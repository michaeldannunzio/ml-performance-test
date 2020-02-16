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

#define FILEPATH "../plasma.csv"

typedef std::map<std::string, std::vector<std::string> > Dataframe;

// global variables
Dataframe df;

// function prototypes
arma::Mat<float> read_csv(std::string&);
void display(Dataframe);

int main(int argc, char *argv[]) {
  std::string s(FILEPATH);

  std::cout << std::endl << "---------------" << std::endl;
  read_csv(s);
  std::cout << "---------------" << std::endl;
  
  return EXIT_SUCCESS;
}

void display(Dataframe df) {
  std::vector<std::string> attrs;

  for (
    Dataframe::iterator it = df.begin();
    it != df.end();
    it++
  ) {
    attrs.push_back(it->first);
  }


}

arma::Mat<float> read_csv(std::string& filepath) {
  arma::Mat<float> mat;

  std::ifstream fin;
  std::vector<std::string> attrs;
  char buffer[256];
  std::istringstream iss;
  std::string value;
  int i = 0;

  fin.open(filepath);

  if (!fin) {
    std::cout << "[ERROR] Unable to open file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }

  while (fin.get(buffer, 256, ',')) {
    value = buffer;
    attrs.push_back(value);
  }

  while (fin.getline(buffer, 256, ',')) {
    value = buffer;
    df[attrs[i]].push_back(value);
    i++;
  }

  fin.close();

  return mat;
}

std::vector<std::string> split(const std::string& line, const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  
  do {
    pos = line.find(delimiter, prev);
    if (pos == std::string::npos) {
      pos = line.length();
    }

    std::string token = line.substr(prev, pos-prev);
    if (!token.empty()) {
      tokens.push_back(token);
    }
    
    prev = pos + delimiter.length();
  } while (pos < line.length() && prev < line.length());
  
  return tokens;
}
