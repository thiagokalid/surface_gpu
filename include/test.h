// test.h
#ifndef TEST_H
#define TEST_H

class Test {
public:
    // Constructor
    Test();

    // Destructor (optional but good practice)
    ~Test();

    // Method to print a message
    void printMessage();

    // Method to get a number
    int getNumber() const;

    // Method to set a number
    void setNumber(int num);

private:
    int number;
};

#endif // TEST_H
