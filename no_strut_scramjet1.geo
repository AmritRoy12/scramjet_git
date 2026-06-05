//Domain dimensions:
Pi = 3.14159265359;
DTR = Pi/180;
Angle_1 = 10;
Angle_2 = 20;
L = 11.0;        // Total length
H = 1.5;         // Total height
//Points:
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {6, 0, 0, 1.0};
Point(4) = {8.51, 0, 0, 1.0};
Point(5) = {11, 0, 0, 1.0};
Point(6) = {11, 2.4284, 0, 1.0};
Point(7) = {6, 0.618, 0, 1.0};
Point(8) = {1, 1.5, 0, 1.0};
Point(9) = {0, 1.5, 0, 1.0};
//Boundary Lines:
Line(1) = {1, 2};    
Line(2) = {2, 3};    
Line(3) = {3, 4};    
Line(4) = {4, 5};    
Line(5) = {5, 6};    
Line(6) = {6, 7};    
Line(7) = {7, 8};    
Line(8) = {8, 9};     
Line(9) = {9, 1};    
//Curve Loop:
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//Plane Surface:
Plane Surface(1) = {1};
//Mesh Control:
Mesh.Algorithm = 8;               // Frontal-Delaunay for quads
Mesh.RecombineAll = 1;            // Recombine triangles into quads
Mesh.RecombinationAlgorithm = 1;  // Blossom recombination (best quality)
Mesh.CharacteristicLengthMin = 0.05;
Mesh.CharacteristicLengthMax = 0.15;

// Physical Groups for Trixi.jl
// Inflow (supersonic freestream) — Lines 9, 8
Physical Curve("Left") = {9};
Physical Curve("TopInlet") = {8};

// Outflow (supersonic) — Lines 1, 2, 3, 5
Physical Curve("Right") = {1, 2, 3, 5};

// Inviscid slip wall — Lines 4
Physical Curve("Bottom") = {4};

// Inviscid slip wall — Lines 6, 7
Physical Curve("Top") = {6, 7};

// Fluid domain
Physical Surface("Fluid") = {1};
