#include <cstdlib>    // C standard library
#include <iostream>   // I/O streaming
#include <vector>     // better than arrays
#include <map>        // for simulated dataframe
#include <fstream>    // file streaming
#include <cmath>      // math functions
#include <string>     // string utilities
#include <sstream>
#include <armadillo>
#include <chrono>

// preprocessor macros
#define FILEPATH "../plasma.csv"

// type aliases
typedef std::map<std::string, std::vector<std::string> > Dataframe;

// global variables
Dataframe df;

// function prototypes
arma::Mat<float> read_csv(std::string&);
std::vector<std::string> split(const std::string&, const std::string&);
std::string strip(const std::string&, const std::string&);
void display(Dataframe);

// entry point
int main(int argc, char *argv[]) {
  std::string s(FILEPATH);

  read_csv(s);
  display(df);
  
  return EXIT_SUCCESS;
}

void display(Dataframe df) {
  std::vector<std::string> attrs;
  std::string attr;

  for (
    Dataframe::iterator it = df.begin();
    it != df.end();
    it++
  ) {
    attr = it->first;
    attrs.push_back(attr);
    std::cout << attr << ",";
  }

  int len = df[attrs[0]].size();
  std::cout << std::endl;

  for (int i = 1; i < len; i++) {
    for (int j = 0; j < attrs.size(); j++) {
      std::cout << df[attrs[j]][i] << ",";
    }
    std::cout << std::endl;
  }
}

arma::Mat<float> read_csv(std::string& filepath) {
  arma::Mat<float> mat;
  std::ifstream fin;
  char buffer[256];
  std::vector<std::string> attrs;
  std::vector<std::string> vals;
  int cols = 0;

  fin.open(filepath);

  if (!fin) {
    std::cout << "[ERROR] Unable to open file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }

  fin.getline(buffer, 256, '\n');  // get column names
  attrs = split(buffer, ",");
  cols = attrs.size();

  for (int i = 0; i < cols; i++) {
    std::cout << attrs[i] << std::endl;
    attrs[i] = strip(attrs[i], "\"");
    std::cout << attrs[i] << std::endl;
  }

  // read remaining data
  while (fin.getline(buffer, 256, '\n')) {
    vals = split(buffer, ",");
    
    for (int i = 1; i < cols; i++) {
      std::string current_attr = attrs[i];

      if (current_attr == "ESR")
        vals[i] = vals[i][5] == '>' ? "1" : "0";

      df[attrs[i]].push_back(strip(vals[i], "\""));
    }
  }

  fin.close();

  return mat;
}

std::vector<std::string> split(const std::string& str, const std::string& delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) {
      pos = str.length();
    }

    std::string token = str.substr(prev, pos-prev);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  
  return tokens;
}

std::string strip(const std::string& str, const std::string& delim) {
  std::vector<std::string> str_arr = split(str, delim);

  std::string final_str = "";

  for (std::string s : str_arr)
    final_str += s;

  return final_str;
}


std::vector<float> sigmoid(std::vector<float> z) {
  std::vector<float> f;
  float fi;

  for (float zi : z) {
    fi = 1 / ( 1 + exp(zi) );
    f.push_back(fi);
  }

  return f;
}
