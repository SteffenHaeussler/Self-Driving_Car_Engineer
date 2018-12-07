#include <deque>
#include <numeric>
#include <math.h>
#include <string>
#include <vector>

#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

    Kp_ = Kp;
    Ki_ = Ki;
    Kd_ = Kd;

    // sKp_ = Kp;
    // sKi_ = Ki;
    // sKd_ = Kd;

    for (unsigned int i = 0; i < 3; ++i){
        parameters.push_back(0.0);
        // speed_parameters.push_back(0.0);
        // dp.push_back(1.0);
    }

    // general_error = 1.0;

}

void PID::UpdateError(double cte) {

    // i_error = cte - p_error;
    // d_error += cte;
    // p_error = cte;

    parameters[1] = cte - parameters[0];
    parameters[2] += cte;
    parameters[0] = cte;
}

double PID::TotalError() {

    // double steer = -Kp_ * p_error - Ki_ * i_error - Kd_ * d_error;

    double steer = -Kp_ * parameters[0] - Ki_ * parameters[1] - Kd_ * parameters[2];

    if (steer > 1.0){
        steer = 1.0;
    }
    else if (steer < -1.0){
        steer = -1.0;
    }

    return steer;
}


// void PID::ThrottleUpdateError(double speed) {

//     // i_error = cte - p_error;
//     // d_error += cte;
//     // p_error = cte;

//     speed_parameters[1] = speed - speed_parameters[0];
//     speed_parameters[2] += speed;
//     speed_parameters[0] = speed;

//     // if (previous_cte.size() >= 100){
//     // previous_cte.pop_front();
//     // }
//     // previous_cte.push_back(cte*cte);

//     // general_error = accumulate( previous_cte.begin(), previous_cte.end(), 0.0)/previous_cte.size();

// }

// double PID::ThrottleTotalError() {

//     // double steer = -Kp_ * p_error - Ki_ * i_error - Kd_ * d_error;

//     double throttle = -sKp_ * speed_parameters[0] - sKi_ * speed_parameters[1] - sKd_ * speed_parameters[2];

//     if (throttle > 1.0){
//         throttle = 1.0;
//     }
//     else if (throttle < -1.0){
//         throttle = -1.0;
//     }

//     return throttle;
// }
