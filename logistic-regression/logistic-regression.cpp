#include <cstdlib>    // C standard library
#include <iostream>   // I/O streaming
#include <vector>     // better than arrays
#include <map>        // key value store
#include <fstream>    // file streaming
#include <cmath>      // math functions
#include <string>     // string utilities
#include <armadillo>  // matrix operations
#include <chrono>     // time

using namespace std;
using namespace arma;
using namespace chrono;

// preprocessor macros
#define FILEPATH "./plasma.csv"

// type aliases
typedef map<string, vector<double> > Dataframe;

// function prototypes
mat sigmoid(mat z);
Dataframe read_csv(string);
vector<string> split(const string&, const string&);
string strip(const string&, const string&);

// entry point
// main logic
int main(int argc, char *argv[]) {
  Dataframe df = read_csv(FILEPATH);

  double learning_rate = 0.001;

  mat weights(2, 1, fill::ones);          // one filled column-vector [2, 1]
  mat labels(df["ESR"]);                  // ESR column vector (one hot encoded values) [32, 1]
  mat prob_vector(32, 1, fill::zeros);    // zero filled column vector [32, 1]
  mat error(32, 1);                       // error/cost column vector [32, 1]
  mat data_matrix = join_rows(            // [32, 2] matrix - first is ones, second fibrinogen
    mat(32, 1, fill::ones),
    mat(df["fibrinogen"])
  );

  time_point<system_clock> startTime;
  time_point<system_clock> endTime;
  long long elapsedTime;
  
  startTime = system_clock::now();

  // gradient descent
  for (int i = 0; i < 500000; i++) {
    prob_vector = sigmoid(data_matrix * weights);
    error = labels - prob_vector;
    weights = weights + learning_rate * data_matrix.t() * error;
  }

  endTime = system_clock::now();
  elapsedTime = duration_cast<milliseconds>(endTime - startTime).count();

  cout << "Duration (s): " << (elapsedTime / 1000.0) << endl;
  weights.print("weights = ");
  
  return EXIT_SUCCESS;
}

// calculate the logit on a matrix
mat sigmoid(mat z) {
  return (1 / (1 + exp(-z)));
}

Dataframe read_csv(string filepath) {
  Dataframe df;
  ifstream fin(filepath);
  char buffer[256];
  vector<string> attrs;
  vector<string> vals;

  // check if file opened
  if (!fin) {
    cout << "[ERROR] Unable to open file: " << filepath << endl;
    exit(EXIT_FAILURE);
  }

  fin.getline(buffer, 256, '\n');   // read first line of csv file - column names
  attrs = split(buffer, ",");

  // clean strip double quotes from column names
  for (int i = 0; i < attrs.size(); i++)
    attrs[i] = strip(attrs[i], "\"");

  // read remaining data
  while (fin.getline(buffer, 256, '\n')) {
    vals = split(buffer, ",");
    
    for (int i = 1; i < attrs.size(); i++) {

      // one hot encode ESR values
      if (attrs[i] == "ESR")
        vals[i] = vals[i][5] == '>' ? "1" : "0";

      vals[i] = strip(vals[i], "\"");
      df[attrs[i]].push_back(stof(vals[i]));  // cast string value to float and save to dataframe
    }
  }

  fin.close();

  return df;
}

vector<string> split(const string& str, const string& delim) {
  vector<string> tokens;
  size_t prev = 0, pos = 0;
  
  do {
    pos = str.find(delim, prev);
    if (pos == string::npos) {
      pos = str.length();
    }

    string token = str.substr(prev, pos-prev);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  
  return tokens;
}

string strip(const string& str, const string& delim) {
  vector<string> str_arr = split(str, delim);

  string final_str = "";

  for (string s : str_arr)
    final_str += s;

  return final_str;
}
