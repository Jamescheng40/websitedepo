



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
        
            
         const dEoverdwv = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial W_v} $$"
         const dLioverdH = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t}  $$"
    
         const ddoverd1Wv = "$$  \\frac{\\partial L1_1}{\\partial L2_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M; \\; \\frac{\\partial L1_2}{\\partial L2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M \\cdots \\frac{\\partial L1_T}{\\partial L2_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M $$"
    
         const d21overdWv = "$$ \\frac{\\partial L2_1}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\frac{\\partial ( W_{{v}_{11}} \\times h_1 )}{\\partial W_{{v}_{11}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{1M}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{MN}}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MN $$ "
         const d22overdWv = "$$ \\frac{\\partial L2_2}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & \\frac{\\partial ( W_{{v}_{21}} \\times h_1 )}{\\partial W_{{v}_{21}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{22}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{MN}}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{21}}}} & \\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MN}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{21}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{22}}}} & 0 & \\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{2N}}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{MN}}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MN $$ "
         
         const ddoverd1timesd21overdWv = "$$ \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
         const ddoverd12timesd21overdWv = "$$ \\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
         const ddoverd1timesd21overdWvcombine = "$$ \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix} \\ni M \\times MN  $$"
         
         //Backward explaination
         const dEoverdG_backward  = "$$  \\frac{\\partial E}{\\partial g_j} = - \\begin{bmatrix} 1 & 1 & \\cdots & 1  \\end{bmatrix} \\odot \\begin{bmatrix} \\frac{Y_1}{P_1} & 0 & \\cdots & 0 \\\\ 0 & \\frac{Y_2}{P_2}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & \\frac{Y_T}{P_T} \\end{bmatrix} \\odot \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} = \\begin{bmatrix} \\frac{Y_1}{P_1} & \\frac{Y_2}{P_2} & \\cdots & \\frac{Y_T}{P_T} \\end{bmatrix} \\odot \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix}  $$" 
         const dEoverdG_backward_cont = "$$ = - \\begin{bmatrix} ((\\frac{Y_1}{P_1} \\times P_1 \\times (1 - P_1)) +(\\frac{Y_2}{P_2} \\times - P_2 \\times P_1) + \\cdots + (\\frac{Y_T}{P_T} \\times - P_T \\times P_1) ) & ((\\frac{Y_1}{P_1} \\times - P_1 \\times P_2) +(\\frac{Y_2}{P_2} \\times P_2 \\times (1 - P_2)) + \\cdots + (\\frac{Y_T}{P_T} \\times - P_T \\times P_2) ) & \\cdots & ((\\frac{Y_1}{P_1} \\times - P_1 \\times P_T) +(\\frac{Y_2}{P_2} \\times - P_2 \\times P_T) + \\cdots + (\\frac{Y_T}{P_T} \\times P_T \\times (1 - P_T)))  \\end{bmatrix}  $$"
         
         const dEoverdG_backward_cont1 = "$$ = - \\begin{bmatrix} ((Y_1 - Y_1 \\times P_1) +(- Y_2 \\times P_1) + \\cdots + (- Y_T \\times P_1) ) & ((- Y_1 \\times P_2) + (Y_2 - Y_2 \\times P_2) + \\cdots + (- Y_T \\times P_2) ) & \\cdots & ((- Y_1 \\times P_T) + (- Y_2 \\times P_T)  + \\cdots + (Y_T - Y_T \\times P_T)  )  \\end{bmatrix}  $$"
         const dEoverdG_backward_cont2 = "$$ = - \\begin{bmatrix} (Y_1 - Y_1 \\times P_1 - Y_2 \\times P_1 + \\cdots - Y_T \\times P_1 ) & (- Y_1 \\times P_2 + Y_2 - Y_2 \\times P_2 + \\cdots - Y_T \\times P_2 ) & \\cdots & (- Y_1 \\times P_T + - Y_2 \\times P_T  + \\cdots + Y_T - Y_T \\times P_T)   \\end{bmatrix}  $$"
         const dEoverdG_backward_cont3 = "$$ = - \\begin{bmatrix} (Y_1 - (Y_1 + Y_2 + \\cdots + Y_T ) \\times P_1 ) &  (Y_2  - (Y_1 + Y_2 + \\cdots + Y_T ) \\times P_2 ) & \\cdots &  (Y_T - (Y_1 + Y_2 + \\cdots + Y_T ) \\times P_T )   \\end{bmatrix}  $$"
         
         const y1toytsumto1 = "$$ (Y_1 + Y_2 + \\cdots + Y_T ) = 1 $$"

         const dEoverdG_backward_cont4 = "$$ = - \\begin{bmatrix} (Y_1 - P_1 ) &  (Y_2  -  P_2 ) & \\cdots &  (Y_T -  P_T )   \\end{bmatrix}  $$"

         const dEoverdG_backward_cont5 = "$$ =  \\begin{bmatrix} (P_1 - Y_1 ) &  (P_2 - Y_2 ) & \\cdots &  (P_T - Y_T )   \\end{bmatrix}  $$"
         
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
                    {dEoverdG_backward}
                    {dEoverdG_backward_cont}
                    {dEoverdG_backward_cont1}
                    {dEoverdG_backward_cont2}
                    {dEoverdG_backward_cont3}
                    {y1toytsumto1}
                    {dEoverdG_backward_cont4}
                    {dEoverdG_backward_cont5}
                    </MathJax>
                </MathJaxContext>
        </div>
            )

}

export default Mathtest;