/*
 * @Author: your name
 * @Date: 2021-08-05 14:30:16
 * @LastEditTime: 2021-08-05 16:20:57
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \deploy\cc\utils.cc
 */

#include "stdafx.h"
#include "utils.h"

//// Convert vector of double to string (for writing MFCC file output)
//std::string vector_to_string(std::vector<double> vec,
//                             const std::string &delimiter) {
//    std::stringstream vecStream;
//    for (int i = 0; i < vec.size() - 1; i++) {
//        vecStream << vec[i];
//        vecStream << delimiter;
//    }
//    vecStream << vec.back();
//    vecStream << "\n";
//    return vecStream.str();
//}
//
//std::string vector_vector_string(std::vector<std::vector<double>> vec,
//                                 const std::string &delimiter) {
//    std::string s1 = "";
//    std::stringstream vec_stream;
//    for (int i = 0; i < vec.size() - 1; ++i) {
//        s1 = vector_to_string(vec[i], delimiter);
//        vec_stream << s1;
//        vec_stream << "\n";
//    }
//    s1 = vector_to_string(vec.back(), delimiter);
//    vec_stream << s1;
//    // vec_stream << "\n";
//    return vec_stream.str();
//}


/**
 * @description: String split with specific pattern
 * @param str: input string to be split
 * @param vec: split results
 * @param pattern: split delimiter
 * @return 
 */
//void SplitWord(const std::string &str, std::vector<std::string>& vec, const std::string& pattern) {
//    std::string::size_type pos1, pos2;
//	pos1 = 0;
//	pos2 = str.find(pattern);
//	while (std::string::npos != pos2) {
//		vec.push_back(str.substr(pos1, pos2 - pos1));
//		pos1 = pos2 + pattern.size();
//		pos2 = str.find(pattern, pos1);
//	}
//	if (pos1 != str.length()) {
//		vec.push_back(str.substr(pos1));
//	}
//}