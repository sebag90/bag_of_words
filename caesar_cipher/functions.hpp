#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <iostream>
#include <vector>
#include <string>



void show_options();

std::vector <char> create_alphabet(std::vector <char> v, int k);

std::string cipher(std::string input_s, std::vector <char> alphb_real, std::vector <char> alphb_cip);

std::string decipher(std::string input_s, std::vector <char> alphb_real, std::vector <char> alphb_cip);

std::string take_input_string();

int take_input_key();

std::vector<std::string> retrieve_files(std::string path = "./input/");

#endif 