# Covid-19
Simulações de modelos matemáticos para Covid-19 com e sem vacinas.

## Primeiro modelo
\begin{align}
\begin{split}
%\dfrac{ds}{dt}&=\mu -\alpha(1-\theta)si-\mu s+\gamma (1-s-i-s_{ick})-\omega s\\
%\dfrac{ds}{dt}&=\mu+\gamma -\alpha(1-\theta)si-\mu s-\gamma (s+i+s_{ick})-\omega s\\
\dfrac{ds}{dt}&=\mu+\gamma -\alpha(1-\theta)si-(\mu+\gamma+\omega) s -\gamma i -\gamma s_{ick}\\
%
\dfrac{di}{dt}&=\alpha(1-\theta)si-(\beta_1+\beta_2+\mu)i\\
\dfrac{ds_{ick}}{dt}&=\beta_2i-(\beta_3+\mu)s_{ick}
\end{split}
\label{eq:constant-perc-pop-reduced-model}
\end{align}
com os parâmetros ótimos obtidos com o *script* 'Ajuste do Índice de Isolamento.ipynb'
