#ifndef PID_H
#define PID_H

class PID {
public:
  /*


  /*
  * Coefficients
  */
  double Kp_;
  double Ki_;
  double Kd_;

  // double sKp_;
  // double sKi_;
  // double sKd_;

  std::vector<double> parameters;
  // std::vector<double> dp;
  // std::deque<double> previous_cte;
  // std::vector<double> speed_parameters;

  // double general_error;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  /*
  * Find parameters for minimizing error.
  */

  // void ThrottleUpdateError(double cte);


  // * Calculate the total PID error.

  // double ThrottleTotalError();

};

#endif /* PID_H */
