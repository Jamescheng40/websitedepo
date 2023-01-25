
import React from 'react';
import { render } from 'react-dom';
import Chart from './Chart';
import { getData } from "./utils"
import { getData2} from "./utils2.js"
import {  TypeChooser } from "react-stockcharts/lib/helper";
import {  csvParseRows } from  "d3-dsv";
import './main.css';
import {Navigation} from 'react-minimal-side-navigation';
import 'react-minimal-side-navigation/lib/ReactMinimalSideNavigation.css';
import {Link,BrowserRouter,Routes, Route  } from 'react-router-dom';
import Home from './LSTMcollection/LSTMderivation'
import LSTMdEoverdWi from './LSTMcollection/LSTMdEoverdWi'
import LSTMdEoverdWf from './LSTMcollection/LSTMdEoverdWf'
import Mathtest from './LSTMcollection/test'
import LSTMdEoverdWc from './LSTMcollection/LSTMdEoverdWc'

// var fakedata = 
// [{
//     "date": "2010-01-04T05:00:00.000Z",
//     "open": 25.436282332605284,
//     "high": 25.835021381744056,
//     "low": 25.411360259406774,
//     "close": 25.710416,
//     "volume": 38409100,
//     "split": "",
//     "dividend": "",
//     "absoluteChange": "",
//     "percentChange": ""
// }];


const loaddata = (prop,event) => {

	//filereader: https://www.w3docs.com/learn-javascript/file-and-filereader.html
	let fileReader = new FileReader(); 
	fileReader.readAsText(event.target.files[0]); 
	fileReader.onload = function() {

	//custom csv file parameter column from the csv producer in java; API doc:https://github.com/d3/d3-dsv:
	const data = csvParseRows(fileReader.result, (d, i) => {
		
		//Enable this for the raw file produced by the data collection thread
			// return {
			// 	date: +d[0], // convert first column to Date
			// 	close: +d[1],
			// 	rsiavggain: +d[2],
			// 	rsiavglose: +d[3], // convert fourth column to number
			// 	RSgainloseratio: +d[4],
			// 	RSIratio: +d[5],
			// 	volume: +d[6],
			// 	high: +d[7],
			// 	low: +d[8],
			// 	open: +d[9]
			// 	};


		//Enable this for the file produced by analysis thread
			return {
				date: +d[0], // convert first column to Date
				//regular candles
				close: +d[1],
				high: +d[2],
			 	open: +d[3],
				low: +d[5],
				// heiken ashi
				open1: +d[34],
				close1: +d[35],
				high1: +d[36],
				low1: +d[37],
				volume: +d[4],
			 	sigbuyvolstra: +d[6],
			 	sigbuyblestra: +d[7],
				sigsell: +d[8],
			 	sigbuyvolcurlowstra: +d[9],
			 	sigbuyblwlinecurlowstra: +d[10],
			 	volma: +d[11],
			 	volmavariance: +d[12],
			 	RSIratio: +d[13],
			 	buyvol: +d[14],
			 	sellvol: +d[15],
			 	buysellvolratio: +d[16],
			 	sigvolestimatwithrsival: +d[17],
			 	botwedgeplace: +d[18],
			 	blwlineplace: +d[19],
			 	pricema: +d[20],
			 	pricema2: +d[21],
			 	ATRwriteS: +d[22],
			 	ATRtop: +d[23],
			 	ATRlow: +d[24],
			 	ATRbuysignal: +d[25],
			 	ATRsellsignal: +d[26],
			 	STupS: +d[27],
			 	STdownS: +d[28],
			 	STbuy: +d[29],
			 	STsell: +d[30],
			 	STfinalline: +d[31],
			 	STfinalupS: +d[32],
			 	STfinaldownS: +d[33],
				CElongS: +d[38],
				CEShortS: +d[39],
				pricevolS: +d[40],
				sighdvertrsiS: +d[41],
				EMA2S: +d[42]


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
const About = () => (
	<div class="body1">
		<div>
		<h2>About</h2>
		</div>
	</div>
);

const Testforothestuff = () => (
	<div class="body1">
      <Mathtest/>
	</div>
);

const Homeroute = () => (
<div class="body1">

	<Home/>
</div>

);

const LSTMdEoverdWcroute = () => (
	<div class="body2">
	
		<LSTMdEoverdWc/>
	</div>
	
	);

const LSTMdEoverdWiroute = () => (
	<div class="body2">
	
		<LSTMdEoverdWi/>
	</div>
	
	);
	
const LSTMdEoverdWfroute = () => (
	<div class="body2">
	
		<LSTMdEoverdWf/>
	</div>
	
	);

function Sidebarnav(){

		return(
	<div class="sidebar">
	<Navigation
		// you can use your own router's api to get pathname
		activeItemId="/management/members"
		onSelect={({itemId}) => {
			// maybe push to the route

			
			window.location.href = websitedefaultsite + itemId;

			//window.location.href = window.location.href + itemId.slice(1);	

		}}
		items={[
		{
			title: 'LSTM Full derivation',
			itemId: 'LSTMderivation',
			// you can use your own custom Icon component as well
			// icon is optional
			elemBefore: () => <div name="inbox" />,
			subNav: [
				{
					title: 'Gradient E with Respect to Wout',
					itemId: 'LSTMderivation/dEwrtdWo',
				},
				{
					title: 'Gradient E with Respect to Win(input)',
					itemId: 'LSTMderivation/dEwrtdWi',
				},
				{
					title: 'Gradient E with Respect to Wc(internal state)',
					itemId: 'LSTMderivation/dEwrtdWc',
				},
				{
					title: 'Gradient E with Respect to Wforget',
					itemId: 'LSTMderivation/dEwrtdWf',
				},
				{
					title: 'Gradient E with Respect to Wv',
					itemId: 'LSTMderivation/dEwrtdWv',
				},
				],
		},
		{
			title: 'Math Basics',
			itemId: 'mathbasics',
			elemBefore: () => <div name="users" />
		},
		{
			title: 'Stock Market CSV parser for fun',
			itemId: 'fileinput',
			subNav: [
			{
				title: 'Teams',
				itemId: '/management/teams',
			},
			],
		},
		]}
		
	/>
	
	</div>
)

}

  function NotFound() {
	return (
	<div>
	<h1>Oops! You seem to be lost.</h1>
	<p>Here are some helpful links:</p>
	<Link to='/'>Home</Link>
	<Link to='/blog'>Blog</Link>
				<Link to='/contact'>Contact</Link>
	</div>
	)
	}
function Fileinputbtn() {
	return(
	<div class="body1">
	<div>
		<input
			id="fileinput"
			type="file"
			name="file"
			accept=".csv"
			onChange={e => { loaddata(this,e); datachange(this) }}
			
		>
		</input>
		
	</div>
	</div>
	)
}

var websitedefaultsite = window.location.href;
var start = 2
var delimiter = '/'

var tokens2 = websitedefaultsite.split(delimiter).slice(0, start),
websitedefaultsite = tokens2.join(delimiter); // this


class ChartComponent extends React.Component {

	
	componentDidMount() 
	{
		//console.log("from mount" + window.location.href);
		//websitedefaultsite = 
		//this.setState({fakedata});
		// getData().then(data => {
		// 	this.setState({ data })
		// })
	}

	componentDidUpdate() {
		console.log("updated it aa");
	}

	render() {
		// if (this.state == null) {
		// 	return <div>Loading...</div>
		// }
		if (this.state != null)
		{
			return 		

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

			<TypeChooser>{type => <Chart type={type} data={this.state.data} />}</TypeChooser>
		</div>
		
		}
		return (
		
			<BrowserRouter>

		<div>
						<Routes>
							<Route path='/about' element={<><Sidebarnav/><About/></>} />
							<Route path='/fileinput' element={<><Sidebarnav/><Fileinputbtn/></>} />
							<Route path='/LSTMderivation' element={<><Sidebarnav/><Homeroute/></>} />
							<Route path='/LSTMderivation/dEwrtdWi' element={<><Sidebarnav/><LSTMdEoverdWiroute/></>} />
							<Route path='/LSTMderivation/dEwrtdWf' element={<><Sidebarnav/><LSTMdEoverdWfroute/></>} />
							<Route path='/LSTMderivation/dEwrtdWc' element={<><Sidebarnav/><LSTMdEoverdWcroute/></>} />
							
							{/* todo: add the default page here */}
							<Route path='/' element={<><Sidebarnav/><About/></>} />
							<Route path='/mathbasics' element={<><Sidebarnav/><About/></>} />
							<Route path='/test' element={<><Sidebarnav/><Testforothestuff/></>} />
							<Route path='*' element={<NotFound />}/>
						</Routes>

		</div>
		</BrowserRouter>
		)
	}
}

render(
	<ChartComponent />,
	document.getElementById("root")
);

