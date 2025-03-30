#include <stdlib.h> // Required for qsort, size_t
#include <stdio.h>  // For potential debugging

// Comparison function for qsort (for integers)
int compare_ints(const void* a, const void* b) {
    int int_a = *((const int*)a);
    int int_b = *((const int*)b);

    if (int_a < int_b) return -1;
    if (int_a > int_b) return 1;
    return 0;
}

// The sorting function matching the expected signature
// Sorts the array in-place.
void sort_array(int* arr, size_t n) {
    if (arr == NULL || n <= 1) {
        return; // Nothing to sort or invalid input
    }
    qsort(arr, n, sizeof(int), compare_ints);
}

// Optional main function for standalone testing of this file
/*
int main() {
    int test_array[] = {5, 1, 4, 2, 8, 3, 9, 6, 7, 0};
    size_t n = sizeof(test_array) / sizeof(test_array[0]);

    printf("Original array: ");
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", test_array[i]);
    }
    printf("\n");

    sort_array(test_array, n);

    printf("Sorted array:   ");
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", test_array[i]);
    }
    printf("\n");

    return 0;
}
*/
#include <stdlib.h> // Required for qsort, size_t
#include <stdio.h>  // For potential debugging

// Comparison function for qsort (for integers)
int compare_ints(const void* a, const void* b) {
    int int_a = *((const int*)a);
    int int_b = *((const int*)b);

    if (int_a < int_b) return -1;
    if (int_a > int_b) return 1;
    return 0;
}

// The sorting function matching the expected signature
// Sorts the array in-place.
void sort_array(int* arr, size_t n) {
    if (arr == NULL || n <= 1) {
        return; // Nothing to sort or invalid input
    }
    qsort(arr, n, sizeof(int), compare_ints);
}

// Optional main function for standalone testing of this file
/*
int main() {
    int test_array[] = {5, 1, 4, 2, 8, 3, 9, 6, 7, 0};
    size_t n = sizeof(test_array) / sizeof(test_array[0]);

    printf("Original array: ");
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", test_array[i]);
    }
    printf("\n");

    sort_array(test_array, n);

    printf("Sorted array:   ");
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", test_array[i]);
    }
    printf("\n");

    return 0;
}
*/
