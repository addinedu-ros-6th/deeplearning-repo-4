#include <Servo.h>

Servo xServo, yServo;

int xpos = 90, ypos = 90;  // 초기 서보 위치
const int x_center = 320, y_center = 240;  // 화면 중앙 좌표
const int deadzone = 30;  // 서보가 멈추는 영역의 반지름

String inputString = "";         // 입력 문자열을 저장할 변수
bool stringComplete = false;     // 문자열이 완전히 수신되었는지 표시

bool patrol_mode = false;  // Patrol 모드 상태를 저장하는 변수
int patrol_x = 90, patrol_y = 90;  // Patrol 모드에서의 서보 위치
String direction = "right";  // Patrol 모드에서의 이동 방향

unsigned long lastValidInputTime = 0; // 마지막으로 유효한 입력을 받은 시간
bool searching = false;  // 서보가 탐색 모드인지 여부를 저장하는 플래그

int search_x = 90, search_y = 90;  // 탐색 모드에서의 서보 위치
String search_direction = "right";  // 탐색 모드에서의 이동 방향

void setup() {
  Serial.begin(9600);
  xServo.attach(10);
  yServo.attach(9);
  xServo.write(xpos);
  yServo.write(ypos);
  inputString.reserve(50);      // 입력 문자열을 위한 메모리 예약
  lastValidInputTime = millis(); // 마지막 유효 입력 시간 초기화
}

void loop() {
  // 직렬 데이터 수신 및 처리
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      stringComplete = true;
      break;
    }
  }

  if (stringComplete) {
    processInput(inputString);
    inputString = "";
    stringComplete = false;
  }

  // Patrol 모드일 경우 움직임 수행
  if (patrol_mode) {
    patrol_move();
  } else {
    // Patrol 모드가 아닐 경우, 탐색 모드 여부 확인
    if (millis() - lastValidInputTime > 5000) {
      // 5초 이상 유효한 입력이 없을 경우 탐색 모드로 전환
      if (!searching) {
        // 탐색 모드 초기화
        searching = true;
        search_x = xServo.read();
        search_y = yServo.read();
        search_direction = "right";
        Serial.println("탐색 모드 시작");
      }
      // 탐색 움직임 수행
      search_move();
    } else {
      if (searching) {
        Serial.println("탐색 모드 종료, 트래킹 모드로 복귀");
      }
      searching = false;  // 유효한 입력이 있으면 탐색 모드 종료
    }
  }
}

void processInput(String data) {
  data.trim();  // 앞뒤 공백 제거

  static bool prev_patrol_mode = false;  // 이전 모드 상태 저장

  if (data.length() > 0) {
    char mode = data.charAt(0);  // 첫 번째 문자는 모드 표시
    patrol_mode = (mode == 'P'); // 모드에 따라 patrol_mode 설정

    // 모드 변경 여부 확인
    if (patrol_mode != prev_patrol_mode) {
      // 모드가 변경된 경우
      if (patrol_mode) {
        // Patrol 모드로 전환 시
        patrol_x = xServo.read();
        patrol_y = yServo.read();
        direction = "right";  // 방향 초기화
        Serial.println("Patrol 모드로 전환");
      } else {
        // Tracking 모드로 전환 시
        xpos = xServo.read();
        ypos = yServo.read();
        Serial.println("Tracking 모드로 전환");
      }
      // 이전 모드 상태 업데이트
      prev_patrol_mode = patrol_mode;
    }

    data = data.substring(1);  // 모드 문자를 제거하여 나머지 데이터 파싱

    int xIndex = data.indexOf('X');
    int yIndex = data.indexOf('Y');
    int mIndex = data.indexOf('M');

    if (xIndex != -1 && yIndex != -1 && mIndex != -1) {
      int x_mid = data.substring(xIndex + 1, yIndex).toInt();
      int y_mid = data.substring(yIndex + 1, mIndex).toInt();
      int move_amount = data.substring(mIndex + 1).toInt();

      // 디버깅 출력
      Serial.print("Mode: ");
      Serial.print(patrol_mode ? "P" : "T");
      Serial.print(", X: ");
      Serial.print(x_mid);
      Serial.print(", Y: ");
      Serial.print(y_mid);
      Serial.print(", Move: ");
      Serial.println(move_amount);

      // x_mid와 y_mid가 -1이 아닌 경우에만 lastValidInputTime 업데이트
      if (x_mid != -1 && y_mid != -1 && move_amount != 0) {
        // 유효한 입력을 받은 경우, 마지막 유효 입력 시간 업데이트
        lastValidInputTime = millis();
        searching = false;  // 탐색 모드 종료
        // 탐색 모드의 서보 위치 동기화
        search_x = xServo.read();
        search_y = yServo.read();

        if (!patrol_mode) {
          // 트래킹 동작 수행
          tracking_move(x_mid, y_mid, move_amount);
        }
      }
      // x_mid와 y_mid가 -1인 경우(객체가 감지되지 않은 경우), lastValidInputTime을 업데이트하지 않음
    }
  }
}

void tracking_move(int x_mid, int y_mid, int move_amount) {
  // move_amount가 0일 때도 탐색 모드로 전환을 위한 처리
  if (move_amount >= 0) {
    // X 좌표에 따른 Pan 서보 제어
    if (x_mid > x_center + deadzone)
      xpos += move_amount;
    else if (x_mid < x_center - deadzone)
      xpos -= move_amount;

    // Y 좌표에 따른 Tilt 서보 제어
    if (y_mid > y_center + deadzone)
      ypos += move_amount;
    else if (y_mid < y_center - deadzone)  // y_center로 수정
      ypos -= move_amount;

    // 서보 각도 제한
    xpos = constrain(xpos, 20, 160);
    ypos = constrain(ypos, 20, 160);

    // 서보 모터 이동
    xServo.write(xpos);
    yServo.write(ypos);
  }
}

void patrol_move() {
  // Patrol 모드에서 서보 모터가 사각형을 그리며 움직임
  if (direction == "right") {
    patrol_x += 1;
    if (patrol_x >= 140) {
      patrol_x = 140;
      direction = "down";
    }
  } else if (direction == "down") {
    patrol_y += 1;
    if (patrol_y >= 100) {  // Y값 상한을 100으로 변경
      patrol_y = 100;
      direction = "left";
    }
  } else if (direction == "left") {
    patrol_x -= 1;
    if (patrol_x <= 40) {
      patrol_x = 40;
      direction = "up";
    }
  } else if (direction == "up") {
    patrol_y -= 1;
    if (patrol_y <= 80) {  // Y값 하한을 80으로 변경
      patrol_y = 80;
      direction = "right";
    }
  }

  // 서보 모터 이동
  xServo.write(patrol_x);
  yServo.write(patrol_y);

  delay(80);  // 움직임 속도 조절을 위한 지연
}

void search_move() {
  // 탐색 모드에서 서보 모터가 사각형을 그리며 움직임
  if (search_direction == "right") {
    search_x += 1;
    if (search_x >= 140) {
      search_x = 140;
      search_direction = "down";
    }
  } else if (search_direction == "down") {
    search_y += 1;
    if (search_y >= 100) {  // Y값 상한을 100으로 변경
      search_y = 100;
      search_direction = "left";
    }
  } else if (search_direction == "left") {
    search_x -= 1;
    if (search_x <= 40) {
      search_x = 40;
      search_direction = "up";
    }
  } else if (search_direction == "up") {
    search_y -= 1;
    if (search_y <= 85) {  // Y값 하한을 80으로 변경
      search_y = 85;
      search_direction = "right";
    }
  }

  // 서보 모터 이동
  xServo.write(search_x);
  yServo.write(search_y);

  delay(70);  // 움직임 속도 조절을 위한 지연
}
