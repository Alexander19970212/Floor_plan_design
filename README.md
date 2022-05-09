# <p align="center">Automated Floor Plan Design</p>
**Оглавление**
1. [Problem](#Problem)
2. [Proposed solution](#solution)
3. [Automated Floor Plan Design](#auto_plan_design)
4. [Algorithm testing](#tests)
5. [In near plans](#plans)
6. [References](#references)

## <a name="Problem"></a><p align="center">Automated Floor Plan Design Problem</p>

* The problem is to create two-dimensional layouts based on topological, geometrical, functional and esthetic constraints [(1)](#first_reference).
* Because of the combinationally explosive nature of the search problem, it is impossible to search exhaustively to find a solution [(1)](#first_reference).
* The following algorithms were proposed as a solution: Evolution optimization [(2)](#second_reference)-[(3)](#third_reference), Bayesian network [(4)](#forth_reference), Semidefinite optimization, Conves optimization [(5)](#fifth_reference), nonlinear programming model [(6)](#sixth_reference), Simulated annealing [(7)](#sevence_reference), Generative Advesarial Network (GAN) [(8)](#eight_reference).
* The main references is "Hybrid Evolutionary Algorithm applied to Automated Generation", 2019 [(3)](#third_reference).  

<br>
<img src="images/problem.jpg" align="center" alt="Alt text" title="Optional title">
<br><br>

## <a name="sulution"></a><p align="center">The proposed solution: Directed Evolutionary Optimization + Greedy Algorithm</p>

![This is image](images/solution.jpg)
<br><br>

## <a name="auto_plan_design"></a><p align="center">Expectations from the algorithm</p>

* The reverse approach is implemented. Objects are placed first, and then the walls.
* Ability to add new target functions without high cost.
* Problems can be solved: compactness of the plan, rooms and corridors of arbitrary shape, taking into account topological constraints (minimum routes, maximim/minimum distances).

![This is gif](images/auto_plan_design.gif)   
<br><br>

## <a name="tests"></a><p align="center">Testing result</p>

| Test 1 | Test 2 | 
|---|---|
| <img src="images/test_1.gif" align="left"  vspace="5" hspace="5" width=400> | <img src="images/test_2.gif" align="left"  vspace="5" hspace="5" width=400> |


<br><br>
