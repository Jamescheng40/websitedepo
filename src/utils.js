import { tsvParse, csvParse, csv } from  "d3-dsv";
import { timeParse } from "d3-time-format";

function parseData(parse) {
	return function(d) {
		d.date = parse(d.date);
		d.open = +d.open;
		d.high = +d.high;
		d.low = +d.low;
		d.close = +d.close;
		d.volume = +d.volume;

		return d;
	};
}
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

const parseDate = timeParse("%Y-%m-%d");

export function getData() {

	const promiseMSFT = 
	//tsvParse(fakedata, parseData(parseDate));
		 fetch("https://cdn.rawgit.com/rrag/react-stockcharts/master/docs/data/MSFT.tsv")
		 .then(response => response.text())
		 .then(data => tsvParse(data, parseData(parseDate)))
	
		// 	csv("./src/RSIfromcsv1hwithvolopenv1.0.csv")
		//  .row(function(d) { return {key: d.key, value: +d.value}; })
		//  .get(function(error, rows) { console.log(rows); });
	
	return promiseMSFT;
}
