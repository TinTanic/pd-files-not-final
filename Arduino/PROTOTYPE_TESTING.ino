#include <Adafruit_PWMServoDriver.h>

// PWM Driver Configurations
#define SERVOMIN 150
#define SERVOMAX 600
#define SERVO_FREQ 50

// Segregator settings
#define DEFAULT_SEGREGATOR_SERVO_A 80
#define DEFAULT_SEGREGATOR_SERVO_B 80
const int forMetals[2] = {45, 95};
const int forPlastics[2] = {45, 55};
const int forPapers[2] = {135, 95};
const int forOthers[2] = {135, 55};
const uint8_t segregatorServoBottom = 0;
const uint8_t segregatorServoTop = 8;

// Iris settings
const uint8_t irisServo = 9;
const uint16_t CLOSE = 140;
const uint16_t OPEN = 0;

// Loader DC Motor Driver Pins
const uint8_t R_EN = 2;
const uint8_t L_EN = 3;
const uint8_t R_PWM = 5;
const uint8_t L_PWM = 6;

// Serial Communication
const uint8_t bufferSize = 64;
char buffer[bufferSize];
int index = 0;
bool newDataReceived = false;

// IR Sensor and Detection
const uint8_t IR_PIN = 8; // Define the pin for the IR sensor
bool objectDetected = false;


// Initializing the PWM servo driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setServoAngle(uint8_t pin, int angle) {
  uint16_t pulselen = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(pin, 0, pulselen);
}

void setupPWMDriver() {
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  setServoAngle(irisServo, CLOSE);
  setServoAngle(segregatorServoBottom, DEFAULT_SEGREGATOR_SERVO_A);
  setServoAngle(segregatorServoTop, DEFAULT_SEGREGATOR_SERVO_B);
  delay(1000);
}

void setupLoaderDriver() {
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(L_PWM, OUTPUT);
  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);
}

void setupIRSensor() {
  pinMode(IR_PIN, INPUT);
}

void checkIRSensor() {
    int sensorValue = digitalRead(IR_PIN);
    Serial.print("IR Sensor Value: ");
    Serial.println(sensorValue);
    unsigned long currentMillis = millis();

    if (sensorValue == LOW) {
            objectDetected = true;
            stopDCMotor();
            Serial.println("Object detected!");
        }
    }



void runDCMotorBackward() {
    analogWrite(R_PWM, 100); // Assuming these values reverse the motor
    analogWrite(L_PWM, 0);
}



void stopDCMotor() {
  analogWrite(R_PWM, 0); // Stop the motor by setting its speed to 0
  analogWrite(L_PWM, 0);
  digitalWrite(R_EN, LOW);
  digitalWrite(L_EN, LOW);
}

void runDCMotor() {
  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);
  analogWrite(R_PWM, 0); // Run the motor at half speed for demonstration
  analogWrite(L_PWM, 100);
}

void receiveDataFromRaspberryPi() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming byte
    char incomingByte = Serial.read();

    // Check if newline character is received
    if (incomingByte == '\n') {
      // Terminate the string
      buffer[index] = '\0';

      // New data is received
      newDataReceived = true;

      // Process the received string
      Serial.print("Received: ");
      Serial.println(buffer);

      // Reset the buffer index for the next string
      index = 0;

      // Clear the serial buffer after processing each complete message
      clearSerialBuffer();
    } else {
      // Store the incoming byte in the buffer
      buffer[index] = incomingByte;
      index++;

      // Check for buffer overflow
      if (index >= bufferSize) {
        // Buffer overflow occurred, discard the data
        index = 0;
      }
    }
  }
}

void clearSerialBuffer() {
  while (Serial.available() > 0) {
    char temp = Serial.read();
  }
}

void setup() {
  Serial.begin(9600);
  setupLoaderDriver();
  setupPWMDriver();
  setupIRSensor();
}

void loop() {


  checkIRSensor(); // Check the IR sensor for objects
  
  if (!objectDetected) { // Run the DC motor only if no object is detected
    runDCMotor();
  } 
  receiveDataFromRaspberryPi();


  if (newDataReceived) {
    // Check what type of trash is detected and ready the segregator
    if (strcmp(buffer, "AlCan") == 0) {
      // For Metal
      Serial.println("Segregator: For Metals");
      setServoAngle(segregatorServoBottom, forMetals[0]);
      delay(1000);
      setServoAngle(segregatorServoTop, forMetals[1]);
      delay(1000);
      setServoAngle(irisServo, OPEN);
      delay(2000);
      setServoAngle(irisServo, CLOSE);
      delay(1000);
      setServoAngle(segregatorServoBottom, DEFAULT_SEGREGATOR_SERVO_A);
      delay(1000);
      setServoAngle(segregatorServoTop, DEFAULT_SEGREGATOR_SERVO_B);
      delay(1000);
      runDCMotor(); // Consider conditions or delays before restarting the motor

    }
    else if (strcmp(buffer, "PlasticCup") == 0 || strcmp(buffer, "PlasticBottle") == 0) {
      // For Plastic
      Serial.println("Segregator: For Plastics");
      setServoAngle(segregatorServoBottom, forPlastics[0]);
      delay(1000);
      setServoAngle(segregatorServoTop, forPlastics[1]);
      delay(1000);
      setServoAngle(irisServo, OPEN);
      delay(2000);
      setServoAngle(irisServo, CLOSE);
      delay(1000);
      setServoAngle(segregatorServoBottom, DEFAULT_SEGREGATOR_SERVO_A);
      delay(1000);
      setServoAngle(segregatorServoTop, DEFAULT_SEGREGATOR_SERVO_B);
      delay(1000);
      runDCMotor(); // Consider conditions or delays before restarting the motor

    }
    else if (strcmp(buffer, "PaperCup") == 0) {
      // For Paper
      Serial.println("Segregator: For Papers");
      setServoAngle(segregatorServoBottom, forPapers[0]);
      delay(1000);
      setServoAngle(segregatorServoTop, forPapers[1]);
      delay(1000);
      setServoAngle(irisServo, OPEN);
      delay(2000);
      setServoAngle(irisServo, CLOSE);
      delay(1000);
      setServoAngle(segregatorServoBottom, DEFAULT_SEGREGATOR_SERVO_A);
      delay(1000);
      setServoAngle(segregatorServoTop, DEFAULT_SEGREGATOR_SERVO_B);
      delay(1000);
      runDCMotor(); // Consider conditions or delays before restarting the motor

    }
    else {
      // For Others
      Serial.println("Segregator: For Others");
      setServoAngle(segregatorServoBottom, forOthers[0]);
      delay(1000);
      setServoAngle(segregatorServoTop, forOthers[1]);
      delay(1000);
      setServoAngle(irisServo, OPEN);
      delay(2000);
      setServoAngle(irisServo, CLOSE);
      delay(1000);
      setServoAngle(segregatorServoBottom, DEFAULT_SEGREGATOR_SERVO_A);
      delay(1000);
      setServoAngle(segregatorServoTop, DEFAULT_SEGREGATOR_SERVO_B);
      delay(1000);
      runDCMotor(); // Consider conditions or delays before restarting the motor

    }

    // Record the time when the Iris opens
    // irisOpenTime = millis();
    // Serial.println("Opening Iris...");
    // setServoAngle(irisServo, OPEN);

    // Set the time when the Iris should close
    // isIrisOpen = true;


    clearSerialBuffer();

    // After processing the trash, make sure to reset object detection flag
    // and run the DC motor again if needed.
    newDataReceived = false;
    objectDetected = false; // Reset the flag or make it dependent on the current IR sensor state
    runDCMotor(); // Consider conditions or delays before restarting the motor
  }

}
