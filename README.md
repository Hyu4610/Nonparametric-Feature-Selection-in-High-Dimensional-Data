---
output:
  pdf_document: default
  html_document: default
---

# Simulation Code for "A General Framework of Nonparametric Feature Selection in High-Dimensional Data""

The codes in this repository are for the first simulation study (regression problem) to perform feature selection using tensor product kernel as introduced in the paper "A General Framework of Nonparametric Feature Selection in High-Dimensional Data". 
\subsection{Setup Requirements}
\begin{itemize}
\item Matlab
\\
Codes are written in Matlab.
\end{itemize}

\subsection{Code Instructions}
K2fun.m is the kernel function for calculating tensor product kernel given $\boldmath{\lambda}$.

The main function is maincodefun.m, where we iteratively estimate $\boldmath{\lambda}$ as introduced in the Methods Section in the paper.

SimGenerate.m includes the code for generating the simulation data.

S1Sample.m is the example of simulation discussed in Simulation Setting 1 in the paper.

More comments can be found in the code.