import json

def filter_json(input_file, output_file, labels_to_keep):
    """
    JSON 파일에서 특정 라벨만 남기고 나머지 데이터를 제거하는 함수.

    :param input_file: 원본 JSON 파일 경로
    :param output_file: 필터링된 JSON 파일 저장 경로
    :param labels_to_keep: 유지하고 싶은 라벨의 리스트
    """
    # JSON 파일 열기
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 필터링된 데이터 리스트
    filtered_data = []

    for item in data['annotations']:
        if item['label'] in labels_to_keep:
            filtered_data.append(item)

    # 필터링된 데이터를 새로운 JSON 구조로 저장
    filtered_json = {
        'annotations': filtered_data,
        'images': data.get('images', []),
        'categories': data.get('categories', []),
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }

    # 결과를 JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_json, f, ensure_ascii=False, indent=4)

    print(f"필터링이 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")

# 사용 예시
input_file = 'input.json'  # 원본 JSON 파일 경로
output_file = 'output_filtered.json'  # 필터링된 결과를 저장할 JSON 파일 경로
labels_to_keep = ['person', 'car', 'cat']  # 유지하고 싶은 라벨들

filter_json(input_file, output_file, labels_to_keep)