
import React from 'react';
import { render } from 'react-dom';
import Chart from './Chart';
import { getData } from "./utils"
import { getData2} from "./utils2.js"

import {  TypeChooser } from "react-stockcharts/lib/helper";

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

const changeHandler = (event) => {

	console.log("asdfasdf");
	getData().then(data => {
		this.setState({ data })
	})
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
						type="file"
						name="file"
						accept=".csv"
						onChange={() => getData2().then(data => {this.setState({data})})}
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

