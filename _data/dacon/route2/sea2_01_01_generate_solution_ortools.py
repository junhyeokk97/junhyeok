import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
import os

dPATH = './_data/dacon/route2/'

def create_data_model(agv_df, task_df):
    """OR-Tools를 위한 데이터 모델을 생성합니다."""
    locations = task_df[['x', 'y']].values.tolist()
    
    distance_matrix = {}
    for from_node in range(len(locations)):
        distance_matrix[from_node] = {}
        for to_node in range(len(locations)):
            distance = abs(locations[from_node][0] - locations[to_node][0]) + \
                       abs(locations[from_node][1] - locations[to_node][1])
            distance_matrix[from_node][to_node] = distance

    data = {
        'locations': locations,
        'distance_matrix': distance_matrix,
        'demands': task_df['demand'].tolist(),
        'service_times': task_df['service_time'].tolist(),
        'deadlines': task_df['deadline'].tolist(),
        'vehicle_capacities': agv_df['capacity'].tolist(),
        'vehicle_speeds': agv_df['speed_cells_per_sec'].tolist(),
        'vehicle_max_distances': agv_df['max_distance'].tolist(),
        'num_vehicles': len(agv_df),
        'depot': 0
    }
    return data

def main():
    """100개 Task 전체에 대한 최적의 초기 해를 생성합니다."""
    # --- 1. 데이터 로드 및 준비 ---
    try:
        agv_df = pd.read_csv(dPATH+'agv.csv')
        task_df_orig = pd.read_csv(dPATH+'task.csv')
    except FileNotFoundError:
        print("🚨 'agv.csv' 또는 'task.csv' 파일이 현재 폴더에 없습니다!")
        return
        
    depot_info = {'task_id': 'DEPOT', 'x': 0, 'y': 0, 'service_time': 0, 'demand': 0, 'deadline': float('inf')}
    task_df = pd.concat([pd.DataFrame([depot_info]), task_df_orig], ignore_index=True)
    
    data = create_data_model(agv_df, task_df)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # --- 2. 제약 조건 설정 (이전과 동일) ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index, 0, max(data['vehicle_max_distances']), True, dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    for i in range(data['num_vehicles']):
        end_index = routing.End(i)
        distance_dimension.CumulVar(end_index).SetRange(0, data['vehicle_max_distances'][i])

    slowest_speed = min(s for s in data['vehicle_speeds'] if s > 0)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['distance_matrix'][from_node][to_node] / slowest_speed
        service_time = data['service_times'][from_node]
        return int(travel_time + service_time)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, 3000, 6000, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    for loc_idx, deadline in enumerate(data['deadlines']):
        if loc_idx == 0: continue
        index = manager.NodeToIndex(loc_idx)
        time_dimension.CumulVar(index).SetRange(0, int(deadline))
    
    # --- 3. (핵심) Task를 건너뛸 수 있도록 페널티 설정 ---
    # 이 부분이 있어야 100개 Task 문제에서 '탐색 실패' 없이 안정적으로 초기 해를 찾을 수 있습니다.
    penalty = 100000 
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # --- 4. 탐색 및 결과 저장 ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(60) 
    # 30:13761 120:13491 240:13092 480:13555 960:a_11926
    #

    print("OR-Tools로 100개 Task 전체에 대한 최적 초기 해 탐색을 시작합니다 (최대시간 : time_limit)...")
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print("✅ 탐색 성공!")
        initial_solution = {}
        # (이하 결과 파싱 및 JSON 저장 코드는 이전과 동일)
        dropped_tasks = []
        for node in range(1, manager.GetNumberOfNodes()):
            if routing.IsStart(manager.NodeToIndex(node)) or routing.IsEnd(manager.NodeToIndex(node)):
                continue
            if solution.Value(routing.NextVar(manager.NodeToIndex(node))) == manager.NodeToIndex(node):
                 task_id = task_df.iloc[node]['task_id']
                 dropped_tasks.append(task_id)
        for vehicle_id in range(data['num_vehicles']):
            agv_id = agv_df.iloc[vehicle_id]['agv_id']
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0: route.append(task_df.iloc[node_index]['task_id'])
                index = solution.Value(routing.NextVar(index))
            initial_solution[agv_id] = route
        
        print(f"\n할당된 Task 수: {len(task_df_orig) - len(dropped_tasks)} / {len(task_df_orig)}")
        if dropped_tasks:
            print(f"제외된 Task 수: {len(dropped_tasks)} (이 Task들은 다음 단계에서 모두 추가됩니다)")

        output_filename = 'initial_solution.json'
        full_output_path = os.path.join(dPATH, output_filename)
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(initial_solution, f, indent=4)
        print(f"\n✅ 초기 해를 '{output_filename}' 파일로 성공적으로 저장했습니다.")
    else:
        print("❌ 탐색에 실패했습니다.")

if __name__ == '__main__':
    main()