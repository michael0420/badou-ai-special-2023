### Proof the geometric center formula of bilinear interpolation

$$
\begin{align}
&Source\;pixel=M\times M;\;center=\frac{1}{M}\sum_{n=0}^{M-1}(x_n,y_n)=(\frac{M-1}{2},\frac{M-1}{2})\\ 
&Distination\;pixel=N\times N\;center=\frac{1}{N}\sum_{n=0}^{N-1}(x_n,y_n)=(\frac{N-1}{2},\frac{N-1}{2})\\
&Sx=\frac{M}{N}\\
&\because if\; center\; is\;equal, \frac{M-1}{2}+C=(\frac{N-1}{2}+C)\times{\frac{M}{N}}\\
&\Rightarrow C(\frac{N-M}{N})=\frac{M(N-1)-N(M-1)}{2N}\\
&\Rightarrow C(\frac{N-M}{N})=\frac{1}{2}\frac{N-M}{N}\\
&\therefore C=\frac{1}{2}
\end{align}
$$

