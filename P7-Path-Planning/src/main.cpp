#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"

#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

    double closestLen = 100000; //large number
    int closestWaypoint = 0;

    for(int i = 0; i < maps_x.size(); i++)
    {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist = distance(x,y,map_x,map_y);
        if(dist < closestLen)
        {
            closestLen = dist;
            closestWaypoint = i;
        }

    }

    return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

    int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

    double map_x = maps_x[closestWaypoint];
    double map_y = maps_y[closestWaypoint];

    double heading = atan2((map_y-y),(map_x-x));

    double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
    int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

    int prev_wp;
    prev_wp = next_wp-1;
    if(next_wp == 0)
    {
        prev_wp  = maps_x.size()-1;
    }

    double n_x = maps_x[next_wp]-maps_x[prev_wp];
    double n_y = maps_y[next_wp]-maps_y[prev_wp];
    double x_x = x - maps_x[prev_wp];
    double x_y = y - maps_y[prev_wp];

    // find the projection of x onto n
    double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
    double proj_x = proj_norm*n_x;
    double proj_y = proj_norm*n_y;

    double frenet_d = distance(x_x,x_y,proj_x,proj_y);

    //see if d value is positive or negative by comparing it to a center point

    double center_x = 1000-maps_x[prev_wp];
    double center_y = 2000-maps_y[prev_wp];
    double centerToPos = distance(center_x,center_y,x_x,x_y);
    double centerToRef = distance(center_x,center_y,proj_x,proj_y);

    if(centerToPos <= centerToRef)
    {
        frenet_d *= -1;
    }

    // calculate s value
    double frenet_s = 0;
    for(int i = 0; i < prev_wp; i++)
    {
        frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
    }

    frenet_s += distance(0,0,proj_x,proj_y);

    return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
    int prev_wp = -1;

    while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
    {
        prev_wp++;
    }

    int wp2 = (prev_wp+1)%maps_x.size();

    double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
    // the x,y,s along the segment
    double seg_s = (s-maps_s[prev_wp]);

    double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
    double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

    double perp_heading = heading-pi()/2;

    double x = seg_x + d*cos(perp_heading);
    double y = seg_y + d*sin(perp_heading);

    return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  double ego_velocity = 0.0;
  int lane_id = 1;
  int lane_width = 4;
  double max_speed_limit = 49.0;
  double rel_speed_limit = 49.0;
  int security_distance = 50;
  // int security_distance_back = 10;
  int trajectory_distance = 30;
  int num_particle = 50;
  double delta_t = 0.02;


h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,
            &ego_velocity, &lane_id, &lane_width, &max_speed_limit, &rel_speed_limit,
            &security_distance, &trajectory_distance, &num_particle, &delta_t]
            (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // Previous path data given to the Planner
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // Previous path's end s and d values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            json msgJson;

            int prev_size = previous_path_x.size();

            bool free_ride = true;
            bool left_lane_free = true;
            bool right_lane_free = true;
            bool center_lane_free = true;
            // bool change_lane = false;

            for (unsigned int i=0; i < sensor_fusion.size(); ++i){

              // update for each car
              bool vehicle_in_front = false;
              bool vehicle_inside_distance = false;

              float vec_d = sensor_fusion[i][6];
              double vx = sensor_fusion[i][3];
              double vy = sensor_fusion[i][4];
              double vec_speed = sqrt(vx*vx+vy*vy);
              double vec_s = sensor_fusion[i][5];

              vec_s += ((double)prev_size)*delta_t*vec_speed;

              if (vec_s > car_s){
                vehicle_in_front = true;
              }

              if ((vec_s-car_s) < security_distance){
                vehicle_inside_distance = true;
              }

              if ((vec_d > 2+lane_width*lane_id - 2) && (vec_d < 2+lane_width*lane_id + 2)){

                if (vehicle_in_front && vehicle_inside_distance){
                  // if vec in front of ego_vec; adapt speed
                  free_ride = false;
                  rel_speed_limit = vec_speed*2.24;
                }
              }

            // check if neighbor lanes are free
              // if (lane_id == 1){
              if ((vec_d >= 1*lane_width) && (vec_d < 2*lane_width)){
                  if (vehicle_in_front && vehicle_inside_distance){
                    center_lane_free = false;
                  }
              }
              // }

              // if (lane_id == 0){
              if ((vec_d >= 0*lane_width) && (vec_d < 1*lane_width) ){
                  if (vehicle_in_front && vehicle_inside_distance){
                    left_lane_free = false;
                  }
              }

              if (vec_d > 2*lane_width){
                  if (vehicle_in_front && vehicle_inside_distance){
                    right_lane_free = false;
                  }
            }
            }

            if (not free_ride){
              if (ego_velocity > rel_speed_limit){
                ego_velocity -= 0.224;
              }

              if ((lane_id == 0) && center_lane_free){
                lane_id += 1;
                // change_lane = true;
              }
              else if ((lane_id == 2) && center_lane_free){
                lane_id -= 1;
                // change_lane = true;
              }
              else if ((lane_id == 1) && left_lane_free){
                lane_id -= 1;
                // change_lane = true;
              }
                else if ((lane_id == 1) && right_lane_free){
                lane_id += 1;
                // change_lane = true;
              }

            }

            else if(ego_velocity < max_speed_limit){
              ego_velocity += 0.224;
            }
            //should work without, put it for logical consistency
            else if(ego_velocity > max_speed_limit){
              ego_velocity -= 0.224;
            }

            vector <double> way_pts_x;
            vector <double> way_pts_y;

            double last_x = car_x;
            double last_y = car_y;
            double last_yaw = deg2rad(car_yaw);


              if (prev_size > 0){
                    car_s = end_path_s;}

            // get two previous waypoints for spline
            if (prev_size< 2){
               double prev_x = car_x - cos(car_yaw);
               double prev_y = car_y - sin(car_yaw);

              way_pts_x.push_back(prev_x);
              way_pts_y.push_back(prev_y);

              way_pts_x.push_back(last_x);
              way_pts_y.push_back(last_y);

            }
            else{

              last_x = previous_path_x[prev_size-1];
              last_y = previous_path_y[prev_size-1];

              double prev_last_x = previous_path_x[prev_size-2];
              double prev_last_y = previous_path_y[prev_size-2];
              double prev_last_yaw = atan2(last_y-prev_last_y,last_x-prev_last_x);

              way_pts_x.push_back(prev_last_x);
              way_pts_y.push_back(prev_last_y);

              way_pts_x.push_back(last_x);
              way_pts_y.push_back(last_y);

              }

            // get 3 future waypoints for spline
            // with python i would make here a variable
            vector<double> traj_wp_0 = getXY(car_s+(1*trajectory_distance),
                                        (2+lane_width*lane_id), map_waypoints_s,
                                        map_waypoints_x, map_waypoints_y);

            vector<double> traj_wp_1 = getXY(car_s+(2*trajectory_distance),
                                            (2+lane_width*lane_id), map_waypoints_s,
                                        map_waypoints_x, map_waypoints_y);

            vector<double> traj_wp_2 = getXY(car_s+(3*trajectory_distance),
                                        (2+lane_width*lane_id), map_waypoints_s,
                                        map_waypoints_x, map_waypoints_y);

            way_pts_x.push_back(traj_wp_0[0]);
            way_pts_x.push_back(traj_wp_1[0]);
            way_pts_x.push_back(traj_wp_2[0]);

            way_pts_y.push_back(traj_wp_0[1]);
            way_pts_y.push_back(traj_wp_1[1]);
            way_pts_y.push_back(traj_wp_2[1]);

           // adjust car angle
            for(unsigned int i=0; i<way_pts_x.size(); i++){

              double shift_x = way_pts_x[i] - last_x;
              double shift_y = way_pts_y[i] - last_y;

              way_pts_x[i] = (shift_x*cos(0-last_yaw)-shift_y*sin(0-last_yaw));
              way_pts_y[i] = (shift_x*sin(0-last_yaw)+shift_y*cos(0-last_yaw));
            }

            tk::spline s;
            s.set_points(way_pts_x, way_pts_y);

            vector<double> traj_x_vals;
            vector<double> traj_y_vals;

            //add previous calculated paths to reduce calculation costs
            for(unsigned int i=0; i < previous_path_x.size(); i++){

              traj_x_vals.push_back(previous_path_x[i]);
              traj_y_vals.push_back(previous_path_y[i]);

            }

            double target_x = num_particle * delta_t * max_speed_limit;
            double target_y = s(target_x);
            double target_dist =  sqrt(target_x*target_x + target_y*target_y);

            double path_increment = (ego_velocity/2.24)*delta_t*target_x/(target_dist);

            for (int i = 1; i <= num_particle - previous_path_x.size(); i++) {

              double pred_x = i * path_increment;
              double pred_y = s(pred_x);

              double temp_x = pred_x;
              double temp_y = pred_y;

              pred_x = (temp_x*cos(last_yaw)-temp_y*sin(last_yaw));
              pred_y = (temp_x*sin(last_yaw)+temp_y*cos(last_yaw));

              pred_x += last_x;
              pred_y += last_y;

              traj_x_vals.push_back(pred_x);
              traj_y_vals.push_back(pred_y);
            }

            msgJson["next_x"] = traj_x_vals;
            msgJson["next_y"] = traj_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
