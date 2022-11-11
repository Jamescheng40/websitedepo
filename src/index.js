
import React from 'react';
import { render } from 'react-dom';
import Chart from './Chart';
import { getData } from "./utils"
import { getData2} from "./utils2.js"

import {  TypeChooser } from "react-stockcharts/lib/helper";

import { tsvParse, csvParse, csv, csvFormat, csvParseRows } from  "d3-dsv";

import './main.css';

var fakedata = 
[{
    "date": "2010-01-04T05:00:00.000Z",
    "open": 25.436282332605284,
    "high": 25.835021381744056,
    "low": 25.411360259406774,
    "close": 25.710416,
    "volume": 38409100,
    "split": "",
    "dividend": "",
    "absoluteChange": "",
    "percentChange": ""
}];

const loaddata = (prop,event) => {

	//filereader: https://www.w3docs.com/learn-javascript/file-and-filereader.html
	let fileReader = new FileReader(); 
	fileReader.readAsText(event.target.files[0]); 
	fileReader.onload = function() {

	//custom csv file parameter column from the csv producer in java; API doc:https://github.com/d3/d3-dsv:
	const data = csvParseRows(fileReader.result, (d, i) => {
		return {
			date: +d[0], // convert first column to Date
			close: +d[1],
			rsiavggain: +d[2],
			rsiavglose: +d[3], // convert fourth column to number
			RSgainloseratio: +d[4],
			RSIratio: +d[5],
			volume: +d[6],
			high: +d[7],
			low: +d[8],
			open: +d[9]
			};
		});
		
	  	console.log(data);
		prop.setState({data});
		
	  }; 
	  fileReader.onerror = function() {
		alert(fileReader.error);
	  }; 

};
  
const datachange = (prop) => {

	console.log("asdsafasdfa2");

	//getData2().then(data => {prop.setState({data})});

};

class ChartComponent extends React.Component {

	componentDidMount() 
	{

		//this.setState({fakedata});
		getData().then(data => {
			this.setState({ data })
		})
	}

	componentDidUpdate() {
		console.log("updated it aa");
	}

	render() {
		if (this.state == null) {
			return <div>Loading...</div>
		}
		return (
			<div>

				<div class="navbar">
				<a href="#home">Home</a>
				<a href="#news">News</a>
				<div class="dropdown">
					<button class="dropbtn">Dropdown 
					<i class="fa fa-caret-down"></i>
					</button>
					<div class="dropdown-content">
					<a onClick={() => getData().then(data => {this.setState({data})})}>Link 1</a>
					<a onClick={() => getData2().then(data => {this.setState({data})})}>Link 2</a>
					<a >Link 3</a>
					</div>
				</div> 
			</div>
				<div>
					{/* File Uploader */}
					<input
						id="fileinput"
						type="file"
						name="file"
						accept=".csv"
						onChange={e => { loaddata(this,e); datachange(this) }}
						style={{ display: "block", margin: "10px auto" }}
					/>
				</div>

			<TypeChooser>
				{type => <Chart type={type} data={this.state.data} />}
			</TypeChooser>



			</div>
		)
	}
}

render(
	<ChartComponent />,
	document.getElementById("root")
);

