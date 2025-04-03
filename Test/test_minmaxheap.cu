#include "minmaxheap.h"
#include <iostream>


void testMinMaxHeapCUDA() {
    try {
        // Create a heap with a capacity of 10
        MinMaxHeapCUDA heap(10);

        // Insert elements
        std::cout << "Inserting elements into the heap:" << std::endl;
        heap.push(20);
        heap.push(10);
        heap.printHeap();
        heap.push(30);
        heap.printHeap();
        heap.push(5);
        heap.printHeap();
        heap.push(15);
        heap.printHeap();
        heap.push(25);
        heap.printHeap();
        heap.push(31);
        heap.printHeap();
        heap.push(1);
        heap.printHeap();
        int poppedValue;
        // Remove max element
        std::cout << "Popping max element: ";
        heap.deleteMinMax(false, &poppedValue);
        std::cout << poppedValue << std::endl;
        heap.printHeap();

        // Remove min element
        std::cout << "Popping min element: ";
        heap.deleteMinMax(true, &poppedValue);
        std::cout << poppedValue << std::endl;
        heap.printHeap();
        std::cout << "Popping min element: ";
        heap.deleteMinMax(true, &poppedValue);
        std::cout << poppedValue << std::endl;
        heap.printHeap();

        // Insert more elements
        std::cout << "Inserting more elements into the heap:" << std::endl;
        heap.push(35);
        heap.push(100);
        heap.push(-1);
        heap.printHeap();

        // Remove max element again
        std::cout << "Popping max element again: ";
        heap.deleteMinMax(false, &poppedValue);
        std::cout << poppedValue << std::endl;
        heap.printHeap();

    } catch (const std::exception& ex) {
        std::cerr << "Exception occurred: " << ex.what() << std::endl;
    }
}

int main() {
    testMinMaxHeapCUDA();
    return 0;
}
