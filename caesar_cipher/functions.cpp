#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <dirent.h>


void show_options(){
    std::cout   << "Options:\n"
                << "\t-h, --help\t\tshow this\n"
                << "\t-c, --cipher\t\tencode a message\n"
                << "\t-c, --cipher -f\t\tencode files in input folder\n"
                << "\t-d, --decipher\t\tdecode a message\n"
                << "\t-d, --decipher -f\tdecode files in input folder\n"
                << std::endl;
}


std::vector <char> create_alphabet(std::vector <char> v, int k){
    std::vector <char> b;
    for (int i=v.size() - k; i <= v.size() - 1; i++){
        b.push_back(v[i]);
    }
    for (int j = 0; j < v.size() - k; j++){
        b.push_back(v[j]);
    }
    return b;
}


std::string cipher(std::string input_s, std::vector <char> alphb_real, std::vector <char> alphb_cip){
    std::string message;
    for (int i=0; i<input_s.size(); i++){
        if (ispunct(input_s[i]) || isspace(input_s[i]) || isdigit(input_s[i])){
            message.push_back(input_s[i]);
        }
        for (int j=0; j<alphb_real.size(); j++){
            if (tolower(input_s[i]) == alphb_cip[j]){
                if (isupper(input_s[i])){
                    message.push_back(toupper(alphb_real[j]));
                }
                else{
                    message.push_back(alphb_real[j]);
                }
            }
        }
    }
    return message;
}


std::string decipher(std::string input_s, std::vector <char> alphb_real, std::vector <char> alphb_cip){
    std::string message;
    for (int i=0; i<input_s.size(); i++){
        if (ispunct(input_s[i]) || isspace(input_s[i]) || isdigit(input_s[i])){
            message.push_back(input_s[i]);
        }
        for (int j=0; j<alphb_real.size(); j++){
            if (tolower(input_s[i]) == alphb_real[j]){
                if (isupper(input_s[i])){
                    message.push_back(toupper(alphb_cip[j]));
                }
                else{
                    message.push_back(alphb_cip[j]);
                }
            }
        }
    }
    return message;
}

std::string take_input_string(){
    std::string a;
    getline(std::cin, a);
    return a;
}

int take_input_key(){
    int a;
    std::cin>>a;
    while(1){
        if(std::cin.fail() || a < 0 || a > 26){
            std::cin.clear();
            std::cin.ignore(std::numeric_limits <std::streamsize> ::max(),'\n');
            std::cout << "You have entered wrong input\n> ";
            std::cin>>a;
        }
        if(!std::cin.fail() && a > 0 && a < 26){
            break;
        }
    }
    std::cout << "Entered key: " << a << std::endl;
    return a;
}

// create a vector with filenames
std::vector<std::string> retrieve_files(std::string path = "./input/"){
    DIR*    dir;
    dirent* pdir;
    std::vector<std::string> files;

    dir = opendir(path.c_str());

    while (pdir = readdir(dir)) {
        files.push_back(pdir->d_name);
    }
    
    return files;
}