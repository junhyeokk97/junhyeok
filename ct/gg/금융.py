import argparse
import re
import json
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# ---------------------------
# CLI
# ---------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./ct/gg")
    ap.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    ap.add_argument("--test_csv", type=str, default=None)                 # ← None이면 base_dir/test.csv 사용
    ap.add_argument("--sample_submission_csv", type=str, default=None)    # ← None이면 base_dir/sample_submission.csv
    ap.add_argument("--out_csv", type=str, default=None)                  # ← None이면 base_dir/submission_minpost.csv
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max_input_len", type=int, default=2048)
    ap.add_argument("--mc_new_tokens", type=int, default=8)
    ap.add_argument("--gen_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()


# ---------------------------
# Helpers
# ---------------------------
MC_PATTERN = re.compile(r'^\s*(?:\(?\d{1,2}[.)]|[①-⑨]|[A-Da-d][.)]?)\s+')

def is_multiple_choice(text: str) -> bool:
    if not isinstance(text, str): return False
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return sum(1 for ln in lines if MC_PATTERN.match(ln)) >= 2

def extract_question_and_choices(full_text: str):
    lines = [ln.strip() for ln in (full_text or '').splitlines() if ln.strip()]
    q_lines, options = [], []
    for ln in lines:
        (options if MC_PATTERN.match(ln) else q_lines).append(ln)
    return " ".join(q_lines), options


def is_multiple_choice(question_text: str) -> bool:
    if not isinstance(question_text, str):
        return False
    lines = question_text.strip().split("\n")
    # 라인 시작에 숫자가 2개 이상 존재하면 객관식으로 간주
    option_count = sum(line.strip() != "" and line.lstrip().split(" ")[0].isdigit() for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> Tuple[str, List[str]]:
    lines = (full_text or "").strip().split("\n")
    q_lines, options = [], []
    for line in lines:
        token0 = line.strip().split(" ")[0] if line.strip() else ""
        if token0.isdigit():
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    return " ".join(q_lines), options

SYSTEM_BASE = (
    "당신은 한국어를 기본으로 응답하는 금융보안 전문가입니다. "
    "영어 용어는 필요 시 포함할 수 있지만, 중국어는 사용하지 마세요. "
    "항상 간결하고 정확하게 답하세요. "
    "‘모르겠다’, ‘무응답’ 같은 표현은 사용하지 마세요."
)

def build_messages(question: str):
    if is_multiple_choice(question):
        q, options = extract_question_and_choices(question)
        user = (
            "다음 객관식 질문에 대해 보안성과 실무 관점에서 가장 적절한 정답을 고르세요.\n"
            "출력 형식은 **반드시** 아래 JSON 형식을 따르세요.\n\n"
            "형식: {\"answer\":\"<정답 번호만>\"}\n"
            "규칙:\n"
            "1) 한국어로 사고하되, 출력은 JSON만 출력합니다(설명 금지).\n"
            "2) 숫자(예: 2)만 넣습니다. 다른 문자는 금지합니다.\n"
            "3) ‘모르겠다’, ‘무응답’ 등의 표현 금지.\n\n"
            f"질문: {q}\n"
            f"선택지:\n" + "\n".join(options) + "\n\n"
            "이제 JSON으로만 답하세요."
        )
    else:
        user = (
            "다음 주관식 질문에 대해 금융보안 실무자의 관점에서 답하세요.\n"
            "출력 형식은 **반드시** 아래 JSON 형식을 따르세요.\n\n"
            "형식: {\"answer\":\"<두 문장 이내의 답변>\"}\n"
            "규칙:\n"
            "1) 한국어를 기본으로 작성합니다. 영어 용어는 필요 시 포함 가능합니다.\n"
            "2) 두 문장 이내로 간결하게.\n"
            "3) ‘모르겠다’, ‘무응답’ 등의 표현 금지.\n"
            "4) JSON 외의 텍스트(설명/수식/주석)는 출력하지 마세요.\n\n"
            f"질문: {question}\n\n"
            "이제 JSON으로만 답하세요."
        )
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": user},
    ]

def build_bad_words_ids(tokenizer):
    bad_words = ["모르겠", "무응답"]
    ids = []
    for w in bad_words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc)
    return ids or None

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    # 경로 해석 + 존재 확인
    BASE = Path(args.base_dir).resolve()
    test_csv = Path(args.test_csv) if args.test_csv else (BASE / "test.csv")
    sub_csv  = Path(args.sample_submission_csv) if args.sample_submission_csv else (BASE / "sample_submission.csv")
    out_csv  = Path(args.out_csv) if args.out_csv else (BASE / "submission_minpost.csv")

    for p, name in [(test_csv, "test_csv"), (sub_csv, "sample_submission_csv")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # 가속 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 모델/토크나이저 (개방 모델: 토큰 인자 필요 없음)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype if dtype != torch.float32 else None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # flash-attn2 설치 시 "flash_attention_2"
    )
    model.eval()

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    # 데이터
    test = pd.read_csv(str(test_csv))
    questions: List[str] = list(test["Question"])
    bad_ids = build_bad_words_ids(tokenizer)

    preds: List[str] = []
    bs = max(1, args.batch_size)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=(torch.bfloat16 if dtype == torch.bfloat16 else torch.float16)):
        for i in tqdm(range(0, len(questions), bs), desc="Inference (min-post, aya)"):
            qs = questions[i:i+bs]
            texts = [tokenizer.apply_chat_template(build_messages(q), tokenize=False) for q in qs]
        # 배치 내에서 객관식/주관식 분리 실행
        mc_idx  = [k for k, q in enumerate(qs) if is_multiple_choice(q)]
        gen_idx = [k for k, q in enumerate(qs) if k not in mc_idx]

        decoded = [""] * len(qs)

        # 1) 객관식: 결정론(greedy)으로 안정화
        if mc_idx:
            texts_mc = [texts[k] for k in mc_idx]
            batch_mc = tokenizer(
                texts_mc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_len
            ).to(model.device)

            out_mc = model.generate(
                **batch_mc,
                max_new_tokens=args.mc_new_tokens,   # 객관식용 짧은 토큰
                do_sample=False,                      # 랜덤 끔 → 재현성 확보
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=bad_ids,
                no_repeat_ngram_size=0               # 숫자 출력에 불필요한 중복억제 off
            )
            dec_mc = tokenizer.batch_decode(out_mc, skip_special_tokens=True)
            for j, k in enumerate(mc_idx):
                decoded[k] = dec_mc[j]

        # 2) 주관식: 샘플링 허용(간결 답변)
        if gen_idx:
            texts_gen = [texts[k] for k in gen_idx]
            batch_gen = tokenizer(
                texts_gen,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_len
            ).to(model.device)

            out_gen = model.generate(
                **batch_gen,
                max_new_tokens=args.gen_new_tokens,  # 주관식용 길이
                do_sample=True,                      # 랜덤 허용
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=bad_ids,
                no_repeat_ngram_size=3               # 군더더기 반복 억제
            )
            dec_gen = tokenizer.batch_decode(out_gen, skip_special_tokens=True)
            for j, k in enumerate(gen_idx):
                decoded[k] = dec_gen[j]


            for txt, q in zip(decoded, qs):
                try:
                    s, e = txt.rfind("{"), txt.rfind("}")
                    obj = json.loads(txt[s:e+1]) if (s!=-1 and e!=-1 and e>s) else {}
                    ans = str(obj.get("answer","")).strip()
                except Exception:
                    ans = ""

                if is_multiple_choice(q):
                    if ans.isdigit():
                        preds.append(ans)
                    else:
                        # 어디에 있든 숫자 하나 찾기
                        m = re.search(r'(?<!\d)(\d{1,2})(?!\d)', ans or txt)
                        preds.append(m.group(1) if m else "미응답")
                else:
                    preds.append(ans if ans else "미응답")


    # 저장
    submission = pd.read_csv(str(sub_csv))
    
    id_candidates = [c for c in submission.columns if c.lower() not in ["answer"]]
    id_col = None
    for c in id_candidates:
        if c in test.columns:
            id_col = c
            break

    if id_col:
        pred_df = pd.DataFrame({id_col: test[id_col].values, "Answer": preds})
        # sample_submission의 id 순서를 유지
        submission = submission.drop(columns=["Answer"], errors="ignore").merge(pred_df, on=id_col, how="left")
    else:
        # id 키가 없다면 최후 수단: 길이 검사 후 덮어쓰기 (기존 로직)
        if len(submission) != len(preds):
            raise ValueError(f"submission rows({len(submission)}) != preds({len(preds)})")
        submission["Answer"] = preds

    submission.to_csv(str(out_csv), index=False, encoding="utf-8-sig")
    print(f"Saved to: {out_csv}")
    print("MC 개수:", sum(is_multiple_choice(q) for q in questions), "/", len(questions))
    print("미응답 개수:", sum(1 for a in preds if a == "미응답"))
if __name__ == "__main__":
    main()