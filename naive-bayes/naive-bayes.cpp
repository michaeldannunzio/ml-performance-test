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
#define FILEPATH "./titanic_project.csv"

// type aliases
typedef map<string, vector<double> > Dataframe;

// function prototypes
Dataframe read_csv(string);
vector<Dataframe> train_test_split(Dataframe);
vector<string> split(const string&, const string&);
string strip(const string&, const string&);
void display(Dataframe);

// entry point
// main logic
int main(int argc, char *argv[]) {
  Dataframe df = read_csv(FILEPATH);
  display(df);

  vector<Dataframe> df_split = train_test_split(df);
  Dataframe train = df_split[0];
  Dataframe test = df_split[1];

  display(train);
  display(test);

  return EXIT_SUCCESS;
}

vector<Dataframe> train_test_split(Dataframe df) {
  Dataframe train;
  Dataframe test;
  vector<string> attrs;

  for (Dataframe::iterator it = df.begin(); it != df.end(); it++)
    attrs.push_back(it->first);

  for (int i = 0; i < df[attrs[0]].size(); i++)
    if (i < 900)
      for (int j = 0; j < attrs.size(); j++)
        train[attrs[j]].push_back(df[attrs[j]][i]);
    else
      for (int j = 0; j < attrs.size(); j++)
        test[attrs[j]].push_back(df[attrs[j]][i]);

  vector<Dataframe> df_split = { train, test };

  return df_split;
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
    
    for (int i = 1; i < attrs.size(); i++)
      df[attrs[i]].push_back(stof(vals[i]));  // cast string value to float and save to dataframe
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

void display(Dataframe df) {
  vector<string> attrs;
  string attr;

  for (Dataframe::iterator it = df.begin(); it != df.end(); it++) {
    attr = it->first;
    attrs.push_back(attr);
    cout << attr << ",";
  }

  int len = df[attrs[0]].size();
  cout << endl;

  for (int i = 1; i < len; i++) {
    for (int j = 0; j < attrs.size(); j++) {
      cout << df[attrs[j]][i] << ",";
    }
    cout << endl;
  }
}
