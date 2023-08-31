#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <sstream> 
#include <omp.h>
#include <optional>
 
using namespace std;

const int PIXEL_ARRAY_SIZE = 8000;
const double PIXEL_SIZE_UM = 2.5;

// electron class
class Electron {
  public:
    bool active;
    int electron_n;
    int event_n;
    std::optional<int> n_timesteps;
    vector<double> path_xi;
    vector<double> path_yi;
    vector<double> path_zi;
};

// function to split string into vector of strings 
vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str); // Turn the string into a stream.
  string tok;
 
  while(getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
 
  return internal;
}
 

int main(int argc, char **argv) {

  double posx, posy, posz;
  double xi,yi,zi; // the current position of the ionization electron 
  double de, total_e;
  int n_electrons;
  vector<Electron> electrons; 
  vector<int> electrons_at_timestep; // the number of electrons that exist at timestep 't'
  double costheta, phi, step_length;
  int** pixel_array = new int*[PIXEL_ARRAY_SIZE];
  for(int i = 0; i < PIXEL_ARRAY_SIZE; i++)
    pixel_array[i] = new int[PIXEL_ARRAY_SIZE];

  double pixel_size = PIXEL_SIZE_UM; //um in x and y
  double pair_energy = 3.6; // avergage energy to make e-h pair in eV
  double top_of_epi = -6.2; // in um
  double bottom_of_epi = -2.2; // in um for the 4um EPI thinned
  // double bottom_of_epi = -1.2; // in um for the 5um K2 EPI

  double thermal_electron_step = 1.0; // in um the average step of a thermal electron before collision and effective randomization

  std::default_random_engine generator;
  //std::normal_distribution<double> distribution_normal(5.0,2.);
  //std::cout << distribution_normal(generator) << std::endl;
  
  std::exponential_distribution<double> distribution_step(thermal_electron_step);

  std::uniform_real_distribution<double> distribution_costheta(-1.,1.);

  std::uniform_real_distribution<double> distribution_phi(0.,2.0*3.1415926);


  ofstream output_file;
  output_file.open("pixelated_2pt5um_tracks_thinned_4um_back_20k_300keV.txt");

  ofstream e_path_file;
  e_path_file.open("pixelated_2pt5um_tracks_thinned_4um_back_20k_300keV_electron_paths.txt");

  //  output_file << "Writing this to a file.\n";
  output_file << "# " << PIXEL_ARRAY_SIZE << " " << PIXEL_ARRAY_SIZE 
              << " " << PIXEL_ARRAY_SIZE*PIXEL_SIZE_UM << " " << PIXEL_ARRAY_SIZE*PIXEL_SIZE_UM << endl;
  
  // read all the lines on an input file                                                                                                
  
  int read_events = 0;
  string line;

  //  ifstream input_file("../B1-build/run_K2_5um_front_10M_300keV.txt");
  // ifstream input_file("../B1-build/test_20k/e300keV_thinned_4um_back_20k.txt");
  ifstream input_file("e300keV_thinned_4um_back_20k.txt");

  if (input_file.is_open())
    {
      while ( getline (input_file,line) )
        {


          // check if header line or track point

          vector<string> sep = split(line, ' ');
          if (sep[0] == "new_e-") {

            read_events++;
            if (read_events > 1) {
              // write the pixel matrix for previous event
              // completed tracing all electrons now write out and rezero array
              // in sparse mode
              for (int indy=0; indy<PIXEL_ARRAY_SIZE; indy++) {
                for (int indx=0; indx<PIXEL_ARRAY_SIZE; indx++) {
                  if (pixel_array[indy][indx] != 0){
                    output_file << indx << " " << indy << " " << pixel_array[indy][indx] << '\n';
                    // zero array element
                    pixel_array[indy][indx] = 0;
                  } 
                }
              }
              // output_file << total_e << '\n';
              // and rezero the matrix
              total_e = 0.;
            }

            if (read_events % 100 == 0){
              cout << "Event number: " << read_events << '\n';
            }

            // event header has event number, x and y of true position direction and primary energy
            output_file << "EV" << " " << (read_events-1) << " " << sep[1] << " " << sep[2] << " " << sep[4] << " " << sep[5] << '\n';
          } else if (sep[0] == "TEST")  {
            // ignore these lines
          } else {
            // is a point in a track
            // posxyz are the coordinates of the energy deposition in mm, here ocnverted to um
            posx = stod(sep[0])*1000.;
            posy = stod(sep[1])*1000.;
            posz = stod(sep[2])*1000.;
            // the energy lost inthe step in MeV units
            de = stod(sep[3]);
            total_e += de;
            // for now track individual electrons without matrices/binning
            // 
            n_electrons = de*1e6 /pair_energy; 

            // clear array of electrons from last event
            
            int start = electrons.size();
            
            Electron default_e;
            default_e.active = true;
            default_e.path_xi.insert(default_e.path_xi.begin(), posx);
            default_e.path_yi.insert(default_e.path_yi.begin(), posy);
            default_e.path_zi.insert(default_e.path_zi.begin(), posz);

            electrons.resize(start + n_electrons, default_e);

            // the original number of electrons
            int electron_number = n_electrons;
            
            // loop over the ionization electrons produced in this step
            while (n_electrons > 0) {
                           
              //cout << "starting position: " << " " << xi << " " << yi << " " << zi << '\n';
              #pragma omp parallel 
              {
                #pragma omp for
                for (int i = start; i < start + electron_number; i++) {
                  if (electrons[i].active) {
                    xi = electrons[i].path_xi.back(); 
                    yi = electrons[i].path_yi.back(); 
                    zi=electrons[i].path_zi.back(); //initialize

                    step_length = distribution_step(generator);
                    costheta = distribution_costheta(generator);
                    phi = distribution_phi(generator);

                    xi += step_length*cos(phi)*sqrt(1.0-costheta*costheta);
                    yi += step_length*sin(phi)*sqrt(1.0-costheta*costheta);
                    zi += step_length*costheta;

                    if (zi > bottom_of_epi) { // bouncing like a mirror from the EPI substrate interfa
                        zi = bottom_of_epi - (zi-bottom_of_epi);
                        // cout << "REFLECTED BOTTOM position: " << " " << xi << " " << yi << " " << zi << '\n';
                    }

                    electrons[i].path_xi.emplace_back(xi);
                    electrons[i].path_yi.emplace_back(yi);
                    electrons[i].path_zi.emplace_back(zi);

                    //cout << "position: " << " " << xi << " " << yi << " " << zi << '\n';
                    //cout << "costheta: " << costheta << '\n';

                    if (zi <= top_of_epi) {
                      // ionization electron has reached the top of EPI
                      // find the position where it 
                      //cout << "hit the top of the EPI " << " " << n_electrons << '\n';
                      // put in charge matrix, eventually refine to find the crossing point of EPI top 
                      // for now report first position above EPI
                      int ii = int(yi/pixel_size);  //+(PIXEL_ARRAY_SIZE-1)/2; 
                      int jj = int(xi/pixel_size);  //+(PIXEL_ARRAY_SIZE-1)/2;

                      //cout << "ii: " << ii << " jj: " << jj << " n: " << pixel_array[ii][jj] << endl;
                      if (ii > PIXEL_ARRAY_SIZE-1 || jj > PIXEL_ARRAY_SIZE-1 || ii < 0 || jj < 0) {
                        cout << "WARNING, pixel outside of range not added to array: ii: " << ii << " jj: " << jj 
                              << " for yi = " << yi << " and xi = " << xi << endl;

                        electrons[i].active = false;
                      } else {
                        pixel_array[ii][jj]++;

                        // int event_num = read_events - 1;

                        // for (int j = 0; j < electrons[i].path_xi.size(); j++) {
                        //   // event_num and n_electrons create a unique identifier for the ionization electron in the sim
                        //   e_path_file << event_num << n_electrons << " " << j << " " << electrons[i].path_xi[j] << " " << electrons[i].path_yi[j] 
                        //               << " " << electrons[i].path_zi[j] << '\n';
                        // }
                        electrons[i].electron_n = n_electrons;
                        electrons[i].event_n = read_events - 1;
                        electrons[i].active = false;
                        int n_timesteps = electrons[i].path_xi.size();
                        electrons[i].n_timesteps = n_timesteps;

                        #pragma omp critical
                        if (electrons_at_timestep.size() < n_timesteps) {
                          electrons_at_timestep.resize(n_timesteps);
                        }
                        
                        for (int k = 0; k < n_timesteps; k++) {
                          #pragma omp critical
                          electrons_at_timestep[k]++;
                        }
                      }
                      //cout << "ii: " << ii << " jj: " << jj << " n: " << pixel_array[ii][jj] << endl;
                      #pragma omp critical
                      n_electrons--;
                    }
                  }
                }
              }
            }

          }

        }

      // write the pixel matrix for the final event
      if(read_events > 1) {
        for (int indy=0; indy<PIXEL_ARRAY_SIZE; indy++) {
          for (int indx=0; indx<PIXEL_ARRAY_SIZE; indx++) {
            if (pixel_array[indy][indx] != 0){
              output_file << indx << " " << indy << " " << pixel_array[indy][indx] << '\n';
              // zero array element
              pixel_array[indy][indx] = 0;
            } 
          }
        }
      }
      input_file.close();
    }

  e_path_file << "# of e per timestep: ";
  for (int i = 0; i < electrons_at_timestep.size(); i++) {
    e_path_file << electrons_at_timestep[i] << " ";
  }
  e_path_file << '\n';
  e_path_file << "Electron snapshots: \n";

  for (int i = 0; i < electrons_at_timestep.size(); i++) {
    for (int j = 0; j < electrons.size(); j++) {
      if (electrons[j].n_timesteps && i < electrons[j].n_timesteps) {
        e_path_file << j << " " << i << " " 
                    << electrons[j].path_xi[i] << " " << electrons[j].path_yi[i] << " " << electrons[j].path_zi[i] << '\n';

      }
    }
  }
  output_file.close();
  e_path_file.close();

  // Delete the dynamically allocated pixel array
  for(int i = 0; i < PIXEL_ARRAY_SIZE; ++i)
    delete [] pixel_array[i];
  delete [] pixel_array;  
}
