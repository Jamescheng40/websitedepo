import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';

function Mathtest(){
    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };

        //dEoverdWv
            // const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"
        const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"
            
        const dgoverdP_i_final = "$$ \\frac{\\partial g_j}{\\partial P_i} = diag(Y_i/P_i) \\ni T \\times T  $$"
        
        const dPoverdLi_final = "$$\\frac{\\partial P_i}{\\partial L_i} = \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} = P_i \\times (\\delta_{ij} - P_j ) $$"
        const kronica = "$$ \\delta_{ij} = \\begin{cases}1, &         \\text{if } i=j,\\\\0, &  \\text{if } i\\neq j.\\end{cases} $$"
        
            
         const dEoverdwv = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1}  $$"
         const dLioverdH = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t}  $$"
    
         const ddoverd1Wv = "$$  \\frac{\\partial L1_1}{\\partial L2_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M; \\; \\frac{\\partial L1_2}{\\partial L2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M \\cdots \\frac{\\partial L1_T}{\\partial L2_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M $$"
    
         const d21overdWv = "$$ \\frac{\\partial L2_1}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\frac{\\partial ( W_{{v}_{11}} \\times h_1 )}{\\partial W_{{v}_{11}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{1M}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MN $$ "
         const d22overdWv = "$$ \\frac{\\partial L2_2}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & \\frac{\\partial ( W_{{v}_{21}} \\times h_1 )}{\\partial W_{{v}_{21}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{22}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{MN}}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{21}}}} & \\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MN}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{21}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{22}}}} & 0 & \\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{2N}}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{MN}}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MN $$ "
         
         const ddoverd1timesd21overdWv = "$$ \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
         const ddoverd12timesd21overdWv = "$$ \\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
         const ddoverd1timesd21overdWvcombine = "$$ \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix} \\ni M \\times MN  $$"
         
         //Backward explaination
         const dEoverdG_backward  = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} =  \\begin{bmatrix} \\frac{1}{n} & \\frac{1}{n} & \\cdots & \\frac{1}{n}  \\end{bmatrix}  \\odot \\begin{bmatrix} \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_1 - Y_1)^2}{\\partial L_T}} \\\\ \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_1}} & \\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}} & \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_2}} & \\cdots & \\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}  \\end{bmatrix} = \\begin{bmatrix} \\frac{1}{n} & \\frac{1}{n} & \\cdots & \\frac{1}{n}  \\end{bmatrix} \\odot \\begin{bmatrix} 2 \\times (L_1 - Y_1) & 0 & \\cdots & 0 \\\\ 0 & 2 \\times (L_2 - Y_2) & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 2 \\times (L_T - Y_T)  \\end{bmatrix}  $$" 
         const dEoverdG_backward_cont = "$$ =  \\begin{bmatrix} ((\\frac{Y_1}{P_1} \\times P_1 \\times (1 - P_1)) +(\\frac{Y_2}{P_2} \\times - P_2 \\times P_1) + \\cdots + (\\frac{Y_T}{P_T} \\times - P_T \\times P_1) ) & ((\\frac{Y_1}{P_1} \\times - P_1 \\times P_2) +(\\frac{Y_2}{P_2} \\times P_2 \\times (1 - P_2)) + \\cdots + (\\frac{Y_T}{P_T} \\times - P_T \\times P_2) ) & \\cdots & ((\\frac{Y_1}{P_1} \\times - P_1 \\times P_T) +(\\frac{Y_2}{P_2} \\times - P_2 \\times P_T) + \\cdots + (\\frac{Y_T}{P_T} \\times P_T \\times (1 - P_T)))  \\end{bmatrix}  $$"
         
         const dEoverdG_backward_cont1 = "$$ = - \\begin{bmatrix} ((Y_1 - Y_1 \\times P_1) +(- Y_2 \\times P_1) + \\cdots + (- Y_T \\times P_1) ) & ((- Y_1 \\times P_2) + (Y_2 - Y_2 \\times P_2) + \\cdots + (- Y_T \\times P_2) ) & \\cdots & ((- Y_1 \\times P_T) + (- Y_2 \\times P_T)  + \\cdots + (Y_T - Y_T \\times P_T)  )  \\end{bmatrix}  $$"
         const dEoverdG_backward_cont2 = "$$ = - \\begin{bmatrix} (Y_1 - Y_1 \\times P_1 - Y_2 \\times P_1 + \\cdots - Y_T \\times P_1 ) & (- Y_1 \\times P_2 + Y_2 - Y_2 \\times P_2 + \\cdots - Y_T \\times P_2 ) & \\cdots & (- Y_1 \\times P_T + - Y_2 \\times P_T  + \\cdots + Y_T - Y_T \\times P_T)   \\end{bmatrix}  $$"
         const dEoverdG_backward_cont3 = "$$ = \\begin{bmatrix} (\\frac{1}{n} \\times 2 \\times (L_1-Y_1)  ) &  ( \\frac{1}{n} \\times 2 \\times (L_2-Y_2)) & \\cdots &  (\\frac{1}{n} \\times 2 \\times (L_T-Y_T) )   \\end{bmatrix}  $$"
         
         const dEoverdG_backward_cont4 = "$$ = - \\begin{bmatrix} (Y_1 - P_1 ) &  (Y_2  -  P_2 ) & \\cdots &  (Y_T -  P_T )   \\end{bmatrix}  $$"

         const dEoverdG_backward_cont5 = "$$ = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix}  $$"
         

         const powerrule = "$$ \\frac{\\partial x^n}{\\partial x} = n x^{n-1} $$"

         const Wvlater = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial W_v} = \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix} \\ni T \\times MT $$"
         const dEoverdWv = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} h_1 \\times (L_1 - Y_1) & h_2 \\times (L_1 - Y_1) & \\cdots & h_M \\times (L_1 - Y_1) & h_1 \\times (L_2 - Y_2) & h_2 \\times (L_2 - Y_2) & \\cdots & h_M \\times (L_2 - Y_2) & \\cdots & \\cdots & h_1 \\times (L_3 - Y_3) & h_2 \\times (L_3 - Y_3) \\cdots h_M \\times (L_T - Y_T) \\end{bmatrix} \\ni 1 \\times MT $$"
         const dEoverdWvpre = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} (\\frac{1}{n} \\times 2 \\times (L_1-Y_1)  ) &  ( \\frac{1}{n} \\times 2 \\times (L_2-Y_2)) & \\cdots &  (\\frac{1}{n} \\times 2 \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix}  $$"
         const dEoverdWvfinal = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} h_1 \\times (L_1 - Y_1) & h_2 \\times (L_1 - Y_1) & \\cdots & h_M \\times (L_1 - Y_1) \\\\ h_1 \\times (L_2 - Y_2) & h_2 \\times (L_2 - Y_2) & \\cdots & h_M \\times (L_2 - Y_2) \\\\  h_1 \\times (L_3 - Y_3) & h_2 \\times (L_3 - Y_3) & \\cdots & h_M \\times (L_T - Y_T) \\end{bmatrix} \\ni T \\times M $$"

        //Wo
         const dEoverdWo = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  ( \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\underline{\\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_{t-1}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_{t-1}}{\\partial a_{t-1}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot( \\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial h_{t-3}}{\\partial O_{t-3}}  \\odot \\frac{\\partial O_{t-3}}{\\partial d13_{t-3}} \\odot \\frac{\\partial d13_{t-3}}{\\partial d14_{t-3}} \\odot \\frac{\\partial d14_{t-3}}{\\partial W_o} } ) \\oplus \\frac{\\partial h_{t-2}}{\\partial O_{t-2}}  \\odot \\frac{\\partial O_{t-2}}{\\partial d13_{t-2}} \\odot \\frac{\\partial d13_{t-2}}{\\partial d14_{t-2}} \\odot \\frac{\\partial d14_{t-2}}{\\partial W_o} )\\oplus \\odot { \\frac{\\partial h_{t-1}}{\\partial O_{t-1}}  \\odot \\frac{\\partial O_{t-1}}{\\partial d13_{t-1}} \\odot \\frac{\\partial d13_{t-1}}{\\partial d14_{t-1}} \\odot \\frac{\\partial d14_{t-1}}{\\partial W_o} } ) \\oplus { \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} } ) $$" 

         const dEoverdht = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix}  $$"

         const dLoverdht = "$$ \\frac{\\partial Li}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\ni T \\times M $$"


         const dhttoverdOttwo = "$$ \\frac{\\partial h_t}{\\partial O_t} =  \\frac{\\partial \\begin{bmatrix} O_{{t}_1} \\times b_{{t}_1} \\\\ O_{{t}_2} \\times b_{{t}_2} \\\\ \\vdots \\\\ O_{{t}_M} \\times b_{{t}_M} \\end{bmatrix}}{\\partial O_t} = diag(b_{t}) = diag(\\sigma_t(C_t)) \\ni M \\times M $$"
         const dOtoverd13wo = "$$ \\frac{\\partial O_t}{\\partial d13_t} = \\frac{\\partial \\begin{bmatrix} sigmoid(d13_{{t}_1}) \\\\ sigmoid(d13_{{t}_2}) \\\\ \\vdots \\\\ sigmoid(d13_{{t}_M}) \\end{bmatrix}}{\\partial d13_{{t}}} = sigmoid(d13_t) \\times (1 - sigmoid(d13_t)) \\ni M \\times M $$"
         
         const dhtoverWo = "$$ \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} =   diag(\\sigma_t(C_t)) \\odot diag(sigmoid(d13_t) \\times (1 - sigmoid(d13_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix}  $$"

         const dhtoverWopre = "$$ \\begin{bmatrix} x_1 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & x_2 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & \\cdots & x_N \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & x_1 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & x_2 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & \\cdots & x_N \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & \\cdots & \\cdots & x_1 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & x_2 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & \\cdots \\ x_N \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) \\end{bmatrix} $$"

         const dhtoverWocont = "$$= \\begin{bmatrix} x_1 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1))  & x_2 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & \\cdots & x_N \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) \\\\ x_1 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  & x_2 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  & \\cdots & x_N \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & x_2 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & \\cdots & x_N \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) \\end{bmatrix} $$"

         const ddoverd1timesd21overdWocombine1 = "$$ \\frac{\\partial d13}{\\partial W_o} = \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
         const tequal3computehighlighted = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial i_{t-3}} \\odot \\frac{\\partial i_{t-3}}{\\partial d6_{t-3}} \\odot \\frac{\\partial d6_{t-3}}{\\partial d7_{t-3}} \\odot \\frac{\\partial d7_{t-3}}{\\partial W_{i_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial i_{t-2}} \\odot \\frac{\\partial i_{t-2}}{\\partial d6_{t-2}} \\odot \\frac{\\partial d6_{t-2}}{\\partial d7_{t-2}} \\odot \\frac{\\partial d7_{t-2}}{\\partial W_{i_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial i_{t-1}} \\odot \\frac{\\partial i_{t-1}}{\\partial d6_{t-1}} \\odot \\frac{\\partial d6_{t-1}}{\\partial d7_{t-1}} \\odot \\frac{\\partial d7_{t-1}}{\\partial W_{i_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} ) $$"
      
         const dEoverdCt = "$$ \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\odot diag(O_t) \\odot (1 - tanh^2(C_t)) $$"

         const dCtoverdWi = "$$ \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} = 1 \\odot diag(C'_t) \\odot (sigmoid(d5_t) \\times (1 - sigmoid(d5_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} $$"
         const dCtoverdWicont = "$$ \\begin{bmatrix} x_1 \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1))  & x_2 \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1)) & \\cdots & x_N \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1)) \\\\ x_1 \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  & x_2 \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  & \\cdots & x_N \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) & x_2 \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) & \\cdots & x_N \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) \\end{bmatrix} $$"
         
         const tequal3computehighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{ \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"
         const rightmosttermwf = "$$ \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f} = 1 \\odot diag(C_{t-1}) \\odot (sigmoid(d1_t) \\times (1 - sigmoid(d1_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} $$"
         const dCtoverdWfcont = "$$ \\begin{bmatrix} x_1 \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1))  & x_2 \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1)) & \\cdots & x_N \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1)) \\\\ x_1 \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  & x_2 \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  & \\cdots & x_N \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) & x_2 \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) & \\cdots & x_N \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) \\end{bmatrix} $$"
         return (
        
        <div>
            {/* <MathJaxContext config={config} version={3}>
                <MathJax inline>
                {Ftgeneral}
        {itgeneral}
        {Otgeneral}
        {Cdtgeneral}
        {Ctgeneral}
        {htgeneral}
        {Ligeneral}
        {Pigeneral}
        {Eigeneral}
<SyntaxHighlighter language="python" style={dracula}>
                {wiprecode}
            </SyntaxHighlighter>
            <MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dhtoverWocont}

                    </MathJax>
                </MathJaxContext>
                </MathJax>
            </MathJaxContext>
         */}

            <MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdwv}
                    </MathJax>
                </MathJaxContext>

                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                    {/* {dEoverdG_j_final}
                    {dgoverdP_i_final}
                    {dPoverdLi_final}
                    {kronica}
                    {dEoverdwv}
                    {ddoverd1Wv}
                    {d21overdWv}
                    {d22overdWv}
                    {ddoverd1timesd21overdWv}
                    {ddoverd12timesd21overdWv}
                    {ddoverd1timesd21overdWvcombine} */}
                    {powerrule}
                    {dEoverdG_backward}
                    {dEoverdG_backward_cont3}
                    {dEoverdG_backward_cont5}
                    {Wvlater}
                    {dEoverdWvpre}
                    {dEoverdWv}
                    {dEoverdWvfinal}
                    {dEoverdWo}
                    {dEoverdht}
                    {dLoverdht}
                    {dhtoverWo}
                    {dhtoverWopre}
                    {ddoverd1timesd21overdWocombine1}
                    {dhtoverWocont}
                    {tequal3computehighlighted}
                    {dEoverdCt}
                    {dCtoverdWi}
                    {dCtoverdWicont}
                    {tequal3computehighlightedwf}
                    {rightmosttermwf}
                    {dCtoverdWfcont}
                    </MathJax>
                </MathJaxContext>
        </div>
            )

}

export default Mathtest;