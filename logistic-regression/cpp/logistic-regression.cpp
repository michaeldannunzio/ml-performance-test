#include <cstdlib>    // C standard library
#include <iostream>   // I/O streaming
#include <vector>     // better than arrays
#include <map>        // for simulated dataframe
#include <fstream>    // file streaming
#include <cmath>      // math functions
#include <string>     // string utilities
#include <armadillo>
#include <chrono>

using namespace std;
using namespace arma;
using namespace chrono;

// preprocessor macros
#define FILEPATH "../plasma.csv"

// type aliases
typedef map<string, vector<double> > Dataframe;
// struct {}

// function prototypes
Dataframe read_csv(string);
vector<string> split(const string&, const string&);
string strip(const string&, const string&);
mat sigmoid(mat z);

// entry point
int main(int argc, char *argv[]) {
  Dataframe df = read_csv(FILEPATH);

  double accuracy = 0;
  double sensitivity = 0;
  double specificity = 0;
  double learning_rate = 0.001;

  mat weights(2, 1, fill::ones);
  mat labels(df["ESR"]);
  mat prob_vector(32, 1, fill::zeros);
  mat error(32, 1);
  mat data_matrix = join_rows(
    mat(32, 1, fill::ones),
    mat(df["fibrinogen"])
  );

  time_point<system_clock> startTime;
  time_point<system_clock> endTime;
  long long elapsedTime;
  
  startTime = system_clock::now();

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

Dataframe read_csv(string filepath) {
  Dataframe df;
  ifstream fin;
  char buffer[256];
  vector<string> attrs;
  vector<string> vals;
  int cols = 0;

  fin.open(filepath);

  if (!fin) {
    cout << "[ERROR] Unable to open file: " << filepath << endl;
    exit(EXIT_FAILURE);
  }

  fin.getline(buffer, 256, '\n');  // get column names
  attrs = split(buffer, ",");
  cols = attrs.size();

  for (int i = 0; i < cols; i++) {
    attrs[i] = strip(attrs[i], "\"");
  }

  // read remaining data
  while (fin.getline(buffer, 256, '\n')) {
    vals = split(buffer, ",");
    
    for (int i = 1; i < cols; i++) {
      string current_attr = attrs[i];

      if (current_attr == "ESR")
        vals[i] = vals[i][5] == '>' ? "1" : "0";

      vals[i] = strip(vals[i], "\"");

      double temp = stof(vals[i]);

      df[attrs[i]].push_back(temp);
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

mat sigmoid(mat z) {
  return (1 / (1 + arma::exp(-z)));
}
