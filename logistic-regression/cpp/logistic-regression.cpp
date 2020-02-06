#include <cstdlib>    // C standard library
#include <iostream>   // I/O streaming
#include <vector>     // better than arrays
#include <map>        // for simulated dataframe
#include <fstream>    // file streaming
#include <cmath>      // math functions
#include <string>     // string utilities
#include <time.h>
#include <armadillo>

// function prototypes
std::vector<std::string> split(const std::string&, const std::string&);
std::map<std::string, std::vector<float> > read_csv(std::string&);

int main(int argc, char *argv[]) {
  return EXIT_SUCCESS;
}

std::vector<std::string> split(const std::string& line, const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = line.find(delimiter, prev);
    if (pos == std::string::npos) pos = line.length();
    std::string token = line.substr(prev, pos-prev);
    if (!token.empty()) tokens.push_back(token);
    prev = pos + delimiter.length();
  } while (pos < line.length() && prev < line.length());
  
  return tokens;
}

arma::Mat read_csv(std::string& filepath) {
// std::map<std::string, std::vector<float> > read_csv(std::string& filepath) {
  // arma::Mat mat;
  std::map<std::string, std::vector<float> > df;
  std::vector<std::string> col;
  std::vector<std::string> buffer;
  std::string line;
  std::ifstream fin(filepath);

  if (!fin) {
    printf("[ERROR] Unable to open file: %s\n", filepath.c_str());
    exit(EXIT_FAILURE);
  }

  getline(fin, line);
  col = split(line, ",");

  while (getline(fin, line)) {
    buffer = split(line, ",");

    for (int i = 0; i < buffer.size(); i++)
      df[col[i]].push_back(stof(buffer[i]));
  }

  return df;
}
