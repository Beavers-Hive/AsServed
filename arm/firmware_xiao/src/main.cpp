#include <Arduino.h>
#include <SCServo.h>

#include "generated_poses.h"

// Seeed XIAO Bus Servo Adapter の公式配線。別基板ではここだけ変更する。
static constexpr int SERVO_RX_PIN = D7;
static constexpr int SERVO_TX_PIN = D6;
static constexpr uint32_t SERVO_BAUD = 1000000;
static constexpr uint32_t HOST_BAUD = 115200;

static constexpr uint16_t BODY_SPEED = 120;
static constexpr uint16_t GRIPPER_SPEED = 80;
static constexpr uint8_t BODY_ACCEL = 30;
static constexpr uint8_t GRIPPER_ACCEL = 20;
static constexpr int16_t POSITION_TOLERANCE_TICKS = 70;
static constexpr uint32_t POSITION_CHECK_INTERVAL_MS = 200;
static constexpr uint32_t MIN_STEP_TIMEOUT_MS = 3500;
static constexpr uint32_t MAX_STEP_TIMEOUT_MS = 12000;
static constexpr uint32_t TIMEOUT_PER_TICK_MS = 6;
static constexpr size_t COMMAND_BUFFER_SIZE = 96;
static constexpr char BRING_PREFIX[] = "BRING_UTENSIL ";
static constexpr char CALIBRATE_CENTER_PREFIX[] = "CALIBRATE_CENTER ";

HardwareSerial &servoSerial = Serial0;
SMS_STS servos;

enum class Grip : uint8_t { OPEN, CLOSED };
enum class Job : uint8_t { NONE, HOME, FORK, CHOPSTICKS, GOTO };

struct MotionStep {
  const ArmPose *pose;
  Grip grip;
  uint16_t travel_ms;
};

static constexpr MotionStep HOME_STEPS[] = {
    {&POSE_HOME, Grip::OPEN, 1800},
};

static constexpr MotionStep FORK_STEPS[] = {
    {&POSE_HOME, Grip::OPEN, 1000},
    {&POSE_RETREAT, Grip::OPEN, 1300},
    {&POSE_APPROACH_FORK, Grip::OPEN, 1800},
    {&POSE_PICK_FORK, Grip::OPEN, 900},
    {&POSE_PICK_FORK, Grip::CLOSED, 850},
    {&POSE_APPROACH_FORK, Grip::CLOSED, 1000},
    {&POSE_APPROACH_PLACE, Grip::CLOSED, 1800},
    {&POSE_PLACE_UTENSIL, Grip::CLOSED, 900},
    {&POSE_PLACE_UTENSIL, Grip::OPEN, 850},
    {&POSE_APPROACH_PLACE, Grip::OPEN, 900},
    {&POSE_RETREAT, Grip::OPEN, 1300},
    {&POSE_HOME, Grip::OPEN, 1500},
};

static constexpr MotionStep CHOPSTICK_STEPS[] = {
    {&POSE_HOME, Grip::OPEN, 1000},
    {&POSE_RETREAT, Grip::OPEN, 1300},
    {&POSE_APPROACH_CHOPSTICKS, Grip::OPEN, 1800},
    {&POSE_PICK_CHOPSTICKS, Grip::OPEN, 900},
    {&POSE_PICK_CHOPSTICKS, Grip::CLOSED, 850},
    {&POSE_APPROACH_CHOPSTICKS, Grip::CLOSED, 1000},
    {&POSE_APPROACH_PLACE, Grip::CLOSED, 1800},
    {&POSE_PLACE_UTENSIL, Grip::CLOSED, 900},
    {&POSE_PLACE_UTENSIL, Grip::OPEN, 850},
    {&POSE_APPROACH_PLACE, Grip::OPEN, 900},
    {&POSE_RETREAT, Grip::OPEN, 1300},
    {&POSE_HOME, Grip::OPEN, 1500},
};

MotionStep gotoStep = {&POSE_HOME, Grip::OPEN, 1800};

struct Runner {
  Job job = Job::NONE;
  const MotionStep *steps = nullptr;
  size_t count = 0;
  size_t index = 0;
  uint32_t next_check_ms = 0;
  uint32_t deadline_ms = 0;
  int16_t target[6] = {};

  bool busy() const { return job != Job::NONE; }
} runner;

char commandBuffer[COMMAND_BUFFER_SIZE] = {};
size_t commandLength = 0;

const char *jobName(Job job) {
  switch (job) {
    case Job::HOME: return "HOME";
    case Job::FORK: return "fork";
    case Job::CHOPSTICKS: return "chopsticks";
    case Job::GOTO: return "goto";
    default: return "none";
  }
}

bool tickInRange(size_t joint, int16_t tick) {
  return tick >= JOINT_MIN_TICKS[joint] && tick <= JOINT_MAX_TICKS[joint];
}

void setTorque(bool enabled) {
  for (uint8_t id : SERVO_IDS) {
    servos.EnableTorque(id, enabled ? 1 : 0);
  }
}

bool allServosOnline() {
  for (uint8_t id : SERVO_IDS) {
    if (servos.Ping(id) != id || servos.getLastError()) {
      Serial.printf("ERR servo_offline id=%u\n", id);
      return false;
    }
  }
  return true;
}

bool buildTarget(const MotionStep &step, int16_t out[6]) {
  for (size_t i = 0; i < 5; ++i) out[i] = step.pose->body[i];
  out[5] = step.grip == Grip::OPEN ? GRIPPER_OPEN_TICK : GRIPPER_CLOSED_TICK;
  for (size_t i = 0; i < 6; ++i) {
    if (!tickInRange(i, out[i])) {
      Serial.printf("ERR target_out_of_range joint=%u tick=%d\n",
                    static_cast<unsigned>(i + 1), out[i]);
      return false;
    }
  }
  return true;
}

void sendTarget(const int16_t target[6]) {
  uint8_t ids[6];
  int16_t positions[6];
  uint16_t speeds[6];
  uint8_t accels[6];
  for (size_t i = 0; i < 6; ++i) {
    ids[i] = SERVO_IDS[i];
    positions[i] = target[i];
    speeds[i] = i == 5 ? GRIPPER_SPEED : BODY_SPEED;
    accels[i] = i == 5 ? GRIPPER_ACCEL : BODY_ACCEL;
  }
  servos.SyncWritePosEx(ids, 6, positions, speeds, accels);
}

bool targetReached(const int16_t target[6]) {
  for (size_t i = 0; i < 6; ++i) {
    const int current = servos.ReadPos(SERVO_IDS[i]);
    if (current < 0 || abs(current - target[i]) > POSITION_TOLERANCE_TICKS) return false;
  }
  return true;
}

void printTargetDiagnostics(const int16_t target[6]) {
  for (size_t i = 0; i < 6; ++i) {
    const int current = servos.ReadPos(SERVO_IDS[i]);
    const int difference = current < 0 ? -1 : abs(current - target[i]);
    if (current < 0 || difference > POSITION_TOLERANCE_TICKS) {
      Serial.printf("DIAG joint=%u current=%d target=%d diff=%d\n",
                    SERVO_IDS[i], current, target[i], difference);
    }
  }
}

void finishJob(bool ok, const char *reason = nullptr) {
  const Job completed = runner.job;
  runner = Runner{};
  if (ok) {
    if (completed == Job::HOME) Serial.println("OK HOME");
    else if (completed == Job::GOTO) Serial.println("OK GOTO");
    else Serial.printf("OK BRING_UTENSIL %s\n", jobName(completed));
  } else {
    // 到達不能は干渉・過負荷の可能性があるため、押し続けずトルクを切る。
    setTorque(false);
    Serial.printf("ERR %s\n", reason == nullptr ? "motion_failed" : reason);
  }
}

bool commandStep(size_t index) {
  if (index >= runner.count) return false;
  if (!buildTarget(runner.steps[index], runner.target)) {
    finishJob(false, "unsafe_target");
    return false;
  }
  int maximumDifference = 0;
  for (size_t i = 0; i < 6; ++i) {
    const int current = servos.ReadPos(SERVO_IDS[i]);
    if (current < 0) {
      finishJob(false, "servo_read_failed");
      return false;
    }
    maximumDifference = max(maximumDifference, abs(current - runner.target[i]));
  }
  sendTarget(runner.target);
  runner.index = index;
  const uint32_t estimatedTimeout =
      MIN_STEP_TIMEOUT_MS + static_cast<uint32_t>(maximumDifference) * TIMEOUT_PER_TICK_MS;
  const uint32_t timeout = min(MAX_STEP_TIMEOUT_MS,
                               max(static_cast<uint32_t>(runner.steps[index].travel_ms),
                                   estimatedTimeout));
  runner.next_check_ms = millis() + POSITION_CHECK_INTERVAL_MS;
  runner.deadline_ms = millis() + timeout;
  return true;
}

void tickRunner() {
  if (!runner.busy() || static_cast<int32_t>(millis() - runner.next_check_ms) < 0) return;

  if (!targetReached(runner.target)) {
    if (static_cast<int32_t>(millis() - runner.deadline_ms) < 0) {
      runner.next_check_ms = millis() + POSITION_CHECK_INTERVAL_MS;
      return;
    }
    printTargetDiagnostics(runner.target);
    finishJob(false, "pose_not_reached");
    return;
  }

  const size_t next = runner.index + 1;
  if (next >= runner.count) {
    finishJob(true);
    return;
  }
  commandStep(next);
}

template <size_t N>
void startJob(Job job, const MotionStep (&steps)[N]) {
  if (runner.busy()) {
    Serial.println("BUSY");
    return;
  }
  if (!POSE_TABLE_CONFIGURED) {
    Serial.println("ERR poses_not_configured");
    return;
  }
  if (!allServosOnline()) return;
  setTorque(true);
  runner.job = job;
  runner.steps = steps;
  runner.count = N;
  runner.index = 0;
  Serial.printf("OK START %s\n", jobName(job));
  commandStep(0);
}

void printJoints() {
  Serial.print("JOINTS");
  for (uint8_t id : SERVO_IDS) Serial.printf(" %u:%d", id, servos.ReadPos(id));
  Serial.println();
}

void calibrateServoCenter(const char *arguments) {
  unsigned id = 0;
  char confirmation[16] = {};
  if (sscanf(arguments, "%u %15s", &id, confirmation) != 2 ||
      strcmp(confirmation, "CONFIRM") != 0) {
    Serial.println("ERR confirmation_required");
    return;
  }
  if (runner.busy()) {
    Serial.println("BUSY");
    return;
  }
  // グリッパーは開閉値を直接記録するため、中位校正の対象外。
  if (id < 1 || id > 5) {
    Serial.println("ERR invalid_servo_id");
    return;
  }

  setTorque(false);
  const int before = servos.ReadPos(static_cast<uint8_t>(id));
  if (before < 0) {
    Serial.printf("ERR servo_offline id=%u\n", id);
    return;
  }

  // STS3215の中位校正。現在の物理姿勢を内部位置の中央へ対応付ける。
  servos.CalibrationOfs(static_cast<uint8_t>(id));
  delay(250);
  servos.EnableTorque(static_cast<uint8_t>(id), 0);
  delay(50);
  const int after = servos.ReadPos(static_cast<uint8_t>(id));
  if (after < 1900 || after > 2200 || servos.getLastError()) {
    Serial.printf("ERR center_calibration_failed id=%u before=%d after=%d\n",
                  id, before, after);
    return;
  }
  Serial.printf("OK CALIBRATE_CENTER id=%u before=%d after=%d\n", id, before, after);
}

const ArmPose *poseByName(const char *name) {
  if (strcmp(name, "HOME") == 0) return &POSE_HOME;
  if (strcmp(name, "APPROACH_FORK") == 0) return &POSE_APPROACH_FORK;
  if (strcmp(name, "PICK_FORK") == 0) return &POSE_PICK_FORK;
  if (strcmp(name, "APPROACH_CHOPSTICKS") == 0) return &POSE_APPROACH_CHOPSTICKS;
  if (strcmp(name, "PICK_CHOPSTICKS") == 0) return &POSE_PICK_CHOPSTICKS;
  if (strcmp(name, "APPROACH_PLACE") == 0) return &POSE_APPROACH_PLACE;
  if (strcmp(name, "PLACE_UTENSIL") == 0) return &POSE_PLACE_UTENSIL;
  if (strcmp(name, "RETREAT") == 0) return &POSE_RETREAT;
  return nullptr;
}

void startGoto(const char *name) {
  if (runner.busy()) {
    Serial.println("BUSY");
    return;
  }
  if (!POSE_TABLE_CONFIGURED) {
    Serial.println("ERR poses_not_configured");
    return;
  }
  const ArmPose *pose = poseByName(name);
  if (pose == nullptr) {
    Serial.println("ERR unknown_pose");
    return;
  }
  if (!allServosOnline()) return;
  setTorque(true);
  gotoStep = {pose, Grip::OPEN, 1800};
  runner.job = Job::GOTO;
  runner.steps = &gotoStep;
  runner.count = 1;
  Serial.printf("OK START GOTO %s\n", name);
  commandStep(0);
}

void handleCommand(char *line) {
  while (*line == ' ') ++line;
  if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(line, "STATUS") == 0) {
    Serial.printf("STATUS configured=%d busy=%d job=%s\n",
                  POSE_TABLE_CONFIGURED ? 1 : 0, runner.busy() ? 1 : 0, jobName(runner.job));
  } else if (strcmp(line, "READ_JOINTS") == 0) {
    printJoints();
  } else if (strcmp(line, "TORQUE_OFF") == 0 || strcmp(line, "STOP") == 0) {
    runner = Runner{};
    setTorque(false);
    Serial.println("OK TORQUE_OFF");
  } else if (strcmp(line, "TORQUE_ON") == 0) {
    if (!POSE_TABLE_CONFIGURED) Serial.println("ERR poses_not_configured");
    else if (allServosOnline()) {
      setTorque(true);
      Serial.println("OK TORQUE_ON");
    }
  } else if (strncmp(line, CALIBRATE_CENTER_PREFIX,
                     strlen(CALIBRATE_CENTER_PREFIX)) == 0) {
    calibrateServoCenter(line + strlen(CALIBRATE_CENTER_PREFIX));
  } else if (strcmp(line, "HOME") == 0) {
    startJob(Job::HOME, HOME_STEPS);
  } else if (strncmp(line, "GOTO ", 5) == 0) {
    startGoto(line + 5);
  } else if (strncmp(line, BRING_PREFIX, strlen(BRING_PREFIX)) == 0) {
    char dish[32] = {};
    char utensil[20] = {};
    if (sscanf(line + strlen(BRING_PREFIX), "%31s %19s", dish, utensil) != 2) {
      Serial.println("ERR bad_command");
    } else if (strcmp(utensil, "fork") == 0) {
      startJob(Job::FORK, FORK_STEPS);
    } else if (strcmp(utensil, "chopsticks") == 0) {
      startJob(Job::CHOPSTICKS, CHOPSTICK_STEPS);
    } else {
      Serial.println("ERR unknown_utensil");
    }
  } else if (*line != '\0') {
    Serial.println("ERR unknown_command");
  }
}

void pollHost() {
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      commandBuffer[commandLength] = '\0';
      handleCommand(commandBuffer);
      commandLength = 0;
    } else if (commandLength + 1 < COMMAND_BUFFER_SIZE) {
      commandBuffer[commandLength++] = c;
    } else {
      commandLength = 0;
      Serial.println("ERR command_too_long");
    }
  }
}

void setup() {
  Serial.begin(HOST_BAUD);
  servoSerial.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
  servos.pSerial = &servoSerial;
  delay(800);
  setTorque(false);  // 起動時に未確認姿勢へ飛ばさない。
  Serial.printf("READY AS_SERVED configured=%d\n", POSE_TABLE_CONFIGURED ? 1 : 0);
}

void loop() {
  pollHost();
  tickRunner();
  delay(1);
}
