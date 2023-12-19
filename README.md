# Neural Net from Scratch
This is a C# implementation of a simple two layer neural net on the [MNIST data set](http://yann.lecun.com/exdb/mnist/) done completely from scratch with no external libraries. 

Default values of 3000 iterations and a batch size of 100 consistently gets around low 90% accuracy on the test set with minimal training time. 

## Dependencies
- [.NET 7](https://dotnet.microsoft.com/en-us/download/dotnet/7.0)

## Usage
```
dotnet run <batch size> <iterations>

Accuracy: 0.2, Iteration: 0
Accuracy: 0.21, Iteration: 10
Accuracy: 0.26, Iteration: 20
Accuracy: 0.25, Iteration: 30
Accuracy: 0.39, Iteration: 40
...
Accuracy: 0.9, Iteration: 2970
Accuracy: 0.94, Iteration: 2980
Accuracy: 0.87, Iteration: 2990
Calculating accuracy on test data...
Test Data Accuracy: 0.9044

```
If no values are specified for batch size and iterations the program will default to 100 and 3000.
