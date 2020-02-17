###################################################################
# Project 4
# Part 2 -- Naive Bayes
# CS 4375.501
# Michael D'Annunzio & Zain Husain
###################################################################

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
#define TRAIN_TEST_SPLIT_INDEX 900

// type aliases
typedef map<string, vector<double> > Dataframe;

// function prototypes
Dataframe read_csv(string);
tuple<Dataframe, Dataframe> train_test_split(Dataframe, int);
vector<string> split(const string&, const string&);
string strip(const string&, const string&);
void display(Dataframe);

// entry point
// main logic
int main(int argc, char *argv[]) {
  Dataframe df = read_csv(FILEPATH);
  Dataframe train, test;
  tie(train, test) = train_test_split(df, TRAIN_TEST_SPLIT_INDEX);

  // display(df);
  // display(train);
  // display(test);



  return EXIT_SUCCESS;
}

tuple<Dataframe, Dataframe> train_test_split(Dataframe df, int split_index) {
  Dataframe train;
  Dataframe test;
  vector<string> attrs;

  // get column names
  for (Dataframe::iterator it = df.begin(); it != df.end(); it++)
    attrs.push_back(it->first);

  // iterate through dataset
  // split on index 900
  for (int i = 0; i < df[attrs[0]].size(); i++)
    if (i < split_index)
      for (int j = 0; j < attrs.size(); j++)
        train[attrs[j]].push_back(df[attrs[j]][i]);
    else
      for (int j = 0; j < attrs.size(); j++)
        test[attrs[j]].push_back(df[attrs[j]][i]);

  return { train, test };
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
  for (int i = 1; i < attrs.size(); i++)
    attrs[i] = strip(attrs[i], "\"");

  attrs[attrs.size()-1] = attrs[attrs.size()-1].substr(0, attrs[attrs.size()-1].length()-1);

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
