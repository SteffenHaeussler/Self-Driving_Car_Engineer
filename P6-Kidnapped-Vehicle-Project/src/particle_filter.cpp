/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    default_random_engine gen(42);
    normal_distribution<> x_rand(x, std[0]);
    normal_distribution<> y_rand(y, std[1]);
    normal_distribution<> theta_rand(theta, std[2]);

    for (unsigned int i=0; i<num_particles; ++i){
        Particle p;
        p.id = i;
        p.x = x_rand(gen);
        p.y = y_rand(gen);
        p.theta = theta_rand(gen);
        p.weight = 1.0;

        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen(42);
    normal_distribution<> x_noise(0.0, std_pos[0]);
    normal_distribution<> y_noise(0.0, std_pos[1]);
    normal_distribution<> theta_noise(0.0, std_pos[2]);

    if (abs(yaw_rate) < 0.0001) {
            yaw_rate = 0.0001;
        }

    for (unsigned int p=0; p<particles.size(); ++p) {

        double x_hat = particles[p].x;
        double y_hat = particles[p].y;
        double theta = particles[p].theta;
        double theta_hat = theta + yaw_rate*delta_t;

        x_hat += (velocity/yaw_rate) * (sin(theta_hat) - sin(theta)) + x_noise(gen);
        y_hat += (velocity/yaw_rate) * (cos(theta) - cos(theta_hat)) + y_noise(gen);
        theta_hat += theta_noise(gen);

        particles[p].x = x_hat;
        particles[p].y = y_hat;
        particles[p].theta = theta_hat;
    }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // removed, since not needed
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    double num = (1/(2 * M_PI * std_x * std_y));

    // for each particle
    for (unsigned p=0; p<particles.size(); ++p){

        double p_weight = 1.0;

        double p_x = particles[p].x;
        double p_y = particles[p].y;
        double p_theta = particles[p].theta;

        // transform each observation to map coordinates
        // probabliy only works for sparse observations
        for (unsigned obs=0; obs<observations.size(); ++obs){

            double min_dist = sensor_range;
            int nearest_idx = observations.size();
            double lm_map_x;
            double lm_map_y;

            LandmarkObs lm_obs = observations[obs];

            double lm_obs_x = lm_obs.x * cos(p_theta) - lm_obs.y * sin(p_theta) + p_x;
            double lm_obs_y = lm_obs.x * sin(p_theta) + lm_obs.y * cos(p_theta) + p_y;

            // find neareat map landmark
            for (unsigned k=0; k<map_landmarks.landmark_list.size(); ++k){

                Map::single_landmark_s lm_map = map_landmarks.landmark_list[k];

                double d = dist(lm_obs_x, lm_obs_y, lm_map.x_f, lm_map.y_f);

                if (d < min_dist){
                    min_dist = d;
                    nearest_idx = k;

                    lm_map_x = lm_map.x_f;
                    lm_map_y = lm_map.y_f;
                }
            }

            // if no landmark is found, go to next observation
            // else: calculate weight update
            if (nearest_idx == observations.size()) {
                continue;
            }

            double denom = pow(lm_obs_x - lm_map_x, 2)/(2 * std_x * std_x) +
                           pow(lm_obs_y - lm_map_y, 2)/(2 * std_y * std_y);

            p_weight *= (num * exp(-denom));

            particles[p].weight = p_weight;
            weights[p] = p_weight;

        }
    }
}


void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> resample_particles;

    default_random_engine gen(42);
    discrete_distribution<> index_rand(weights.begin(), weights.end());

    for (unsigned i=0; i < num_particles; ++i){

        int idx = index_rand(gen);
        resample_particles.push_back(particles[idx]);
    }
    particles = resample_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}


string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
