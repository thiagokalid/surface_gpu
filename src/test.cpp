// test.cpp
#include "test.h"
#include <iostream>

// Constructor definition
Test::Test() : number(0) {  // Initializing number to 0 by default
    std::cout << "Test object created!" << std::endl;
}

// Destructor definition
Test::~Test() {
    std::cout << "Test object destroyed!" << std::endl;
}

// Method to print a message
void Test::printMessage() {
    std::cout << "Hello from the Test class!" << std::endl;
}

// Method to get the current number
int Test::getNumber() const {
    return number;
}

// Method to set the number
void Test::setNumber(int num) {
    number = num;
}
