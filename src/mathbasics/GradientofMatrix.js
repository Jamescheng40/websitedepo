
import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
import img1 from './gradientofsum.JPG'
import img2 from './JacobianMatrixfromPaper.JPG'


function Gradientmatrix(){

    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };
    
      const ColoredLine = ({ color }) => (
        <hr
            style={{
                color: color,
                backgroundColor: color,
                height: 1
            }}
        />
    );

    const dotproductexplanation = "$$ a \\odot b = \\begin{bmatrix} a_1 & a_2 & \\cdots & a_T \\end{bmatrix} \\odot \\begin{bmatrix} b_1 \\\\ b_2 \\\\ \\vdots \\\\ b_T \\end{bmatrix} = a_1 \\times b_1 + a_2 \\times b_2 + \\cdots + a_T \\times b_T $$"
    const dotproductexplanationcont = "$$ a \\odot b = \\begin{bmatrix} a_1 & a_2 & \\cdots & a_T \\end{bmatrix} \\odot \\begin{bmatrix} b_1 \\\\ b_2 \\\\ \\vdots \\\\ b_T \\end{bmatrix} = a_1 \\times b_1 + a_2 \\times b_2 + \\cdots + a_T \\times b_T = \\sum_{i=0}^T a \\otimes b  $$"
    const diagexample = "$$ diag(x1) = \\begin{bmatrix} x1 & 0 & 0 \\\\ 0 & x1 & 0  \\\\ 0 & 0 & x1 \\end{bmatrix} \\ni 3 \\times 3 $$"
    return (
        <div>
            <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style={{fontSize:32.04}}><strong>Math Basics for deriving Gradient Descent</strong></span></h1>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; There are some great articles such as <a href="https://arxiv.org/pdf/1802.01528.pdf">this</a>, and <a href="https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-ac15e178bbd">this</a>&nbsp;that&nbsp;explained how some of the important Math concepts related to Gradient Descent. Here in this section, we will provide a summary to their conclusion for reference when you are looking at derivation for weight matrix for different papers LSTM, Transformer, and Vanilla.</p>

            <p>&nbsp;</p>


            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; This post will cover the basic math needed for all the LSTM posts you see above. Some equations below will be cropped based on the <a href="http://a">paper</a>.</p>

<p>&nbsp;</p>

<p><span style={{fontSize:26.7}}>Explanation of summation over dot product and their derivative</span></p>

            <ColoredLine color="#E3E3E3" />
            
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; First of all, let us look at why dot-product&nbsp;<strong>⨀&nbsp;</strong>can be replaced with&nbsp;<strong>&sum;a &otimes; b</strong>.&nbsp;&nbsp;The answer is actually simpler than you think, let us look at an example below. Let us assume that <strong>a</strong> is a <strong>1xT</strong> matrix and <strong>b </strong>is a <strong>Tx1</strong> matrix.&nbsp;</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dotproductexplanation}
                </MathJax>
            </MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see above, if you were to multiply the matrix, which you have probably learned from your University Calculus course, the end result is a sum of the product from each element of the matrix. So it makes sense that it is a summation product of row and column. Let us write down the final result.</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dotproductexplanationcont}
                </MathJax>
            </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;To find the derivative is simple. Let us look at what the paper stated.</p>

<p><img src={img1}/></p>

<p>&nbsp;</p>

<p>&nbsp;</p>

<p><span style={{fontSize:26.7}}>Explanation of Matrix Jacobian or partial derivative of a matrix</span></p>

<ColoredLine color="#E3E3E3" />
            
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;When dealing with matrix and their derivative, it will be imperative to know how. Here is the summary from the paper &lt;1802.01528.pdf&gt;.</p>

<p><img src={img2}/></p>

<p>&nbsp;</p>

<p><span style={{fontSize:26.7}}>Some common glossaries used on the derivation page</span></p>

<ColoredLine color="#E3E3E3" />
            
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;You might commonly see diag in the equation, what is diag. Diag is basically an identity matrix except the diagonal has been multiplied with a constant. For example <strong>diag(x1)</strong> with dimension <strong>3 X 3</strong> is basically</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {diagexample}
                </MathJax>
            </MathJaxContext>
            
            <p><span style={{fontSize:"30px"}}><strong>Reference:</strong></span></p>
            <ColoredLine color="#E3E3E3" />

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>1, The Matrix Calculus You Need For Deep Learning <a href="https://arxiv.org/pdf/1802.01528.pdf">link</a></p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>2, How Do You Find the Partial Derivative of a Function <a href="https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-ac15e178bbd">link</a></p>


        </div>
    )
}
  
export default Gradientmatrix;