#include <cstdlib>    // C standard library
#include <iostream>   // I/O streaming
#include <vector>     // better than arrays
#include <map>        // key value store
#include <fstream>    // file streaming
#include <cmath>      // math functions
#include <string>     // string utilities
#include <armadillo>  // matrix operations
#include <ctime>     // time

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
void display(vector<double>);

// entry point
// main logic
int main(int argc, char *argv[]) {
  unsigned int tp = 0;
  unsigned int tn = 0;
  unsigned int fp = 0;
  unsigned int fn = 0;

  double acc;
  double sens;
  double spec;

  clock_t startTime;
  clock_t endTime;

  Dataframe df = read_csv(FILEPATH);

  int age = 0;
  int pclass = 1;
  int sex = 2;
  int survived = 3;

  // make single matrix from typedef Dataframe
  mat data = join_rows(
    mat(df["age"]),
    mat(df["pclass"]),
    mat(df["sex"]),
    mat(df["survived"])
  );

  // train-test split
  mat train = data.rows(0, 900);
  mat test = data.rows(901, data.n_rows - 1);

  startTime = clock();

  vec apriori = {
    mat(train.rows(find(train.col(survived) == 0))).n_rows / (double) train.n_rows,
    mat(train.rows(find(train.col(survived) == 1))).n_rows / (double) train.n_rows
  };

  uvec count_survived = {
    mat(train.rows(find(train.col(survived) == 0))).n_rows,
    mat(train.rows(find(train.col(survived) == 1))).n_rows
  };

  mat lh_pclass(2, 3, fill::zeros);
  for (int sv = 0; sv < 2; sv++) {
    mat S = mat(train.rows(find(train.col(survived) == sv)));
    for (int pc = 0; pc < 3; pc++) {
      lh_pclass(sv, pc) = mat(S.rows(find(S.col(pclass) == pc + 1))).n_rows / (double) count_survived[sv];
    }
  }

  mat lh_sex(2, 2, fill::zeros);
  for (int sv = 0; sv < 2; sv++) {
    mat S = mat(train.rows(find(train.col(survived) == sv)));
    for (int sx = 0; sx < 2; sx++) {
      lh_sex(sv, sx) = mat(S.rows(find(S.col(sex) == sx))).n_rows / (double) count_survived[sv];
    }
  }

  vec age_mean = { 0, 0 };
  vec age_var = { 0, 0 };
  for (int sv = 0; sv < 2; sv++) {
    mat S = mat(train.rows(find(train.col(survived) == sv)));
    age_mean[sv] = mean(S.col(age));
    age_var[sv] = var(S.col(age));
  }

  auto calc_age_lh = [](int _age, double mean_v, double var_v) {
    double x = (1 / sqrt(2 * M_PI * var_v));
    double y = pow((_age - mean_v), 2) / (2 * var_v);
    return x * y;
  };

  auto calc_raw_prob = [lh_pclass, lh_sex, apriori, calc_age_lh, age_mean, age_var](int _pclass, int _sex, int _age) {
    double num_s = (
      lh_pclass(1, _pclass - 1) *
      lh_sex(1, _sex) *
      apriori[1] *
      calc_age_lh(_age, age_mean[1], age_var[1])
    );

    double num_p = (
      lh_pclass(0, _pclass - 1) *
      lh_sex(0, _sex) *
      apriori[0] *
      calc_age_lh(_age, age_mean[0], age_var[0])
    );

    double denominator = (
      lh_pclass(1, _pclass - 1) *
      lh_sex(1, _sex) *
      calc_age_lh(_age, age_mean[1], age_var[1]) *
      apriori[1] +
      lh_pclass(0, _pclass - 1) *
      lh_sex(0, _sex) *
      calc_age_lh(_age, age_mean[0], age_var[0]) *
      apriori[0]
    );

    vec prob_survived = {
      num_s / denominator,
      num_p / denominator
    };

    return prob_survived;
  };

  auto predict = [](double x) {
    return (int)round(x);
  };

  vec raw;

  // testing
  for (int i = 0; i < 146; i++) {
    int _pclass = test.col(pclass)[i];
    int _sex = test.col(sex)[i];
    int _age = test.col(age)[i];

    raw = calc_raw_prob(_pclass, _sex, _age); // col vector is size 2
    int pred = predict(raw[0]);

    if (pred == 0)
      if (test.col(survived)[i] == 0)
        tn += 1;
      else
        fn += 1;
    else
      if (test.col(survived)[i] == 1)
        tp += 1;
      else
        fp += 1;
  }

  acc = (tp + tn) / (double) test.n_rows;
  sens = (double) tp / (tp + fn);
  spec = (double) tn / (tn + fp);

  endTime = clock();

  cout << "Duration (s): " << (((float)endTime - (float)startTime) / CLOCKS_PER_SEC)*1000 << endl;
  cout << "True Positive: " << tp << endl;
  cout << "True Negative: " << tn << endl;
  cout << "False Positive: " << fp << endl;
  cout << "False Negative: " << fn << endl;
  cout << "Accuracy: " << acc << endl;
  cout << "Sensitivity: " << sens << endl;
  cout << "Specificity: " << spec << endl;

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
