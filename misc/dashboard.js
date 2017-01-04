/* =============================== *
 *  Deep Learning Dashboard v0.1   *
 * Jin-Hwa Kim (jnhwkim@snu.ac.kr) *
 *      License: BSD 3-Clause      *
 * =============================== */
var exec = require('child_process').execSync;  // Added in: v0.11.12
function execute(command){
    return exec(command).toString();
};
var csvjson = require('csvjson');
var logfile = process.argv[2] && [process.argv[2]] || getRecentLogs(4);
var start_iter = process.argv[3] || 0
var end_iter = process.argv[4] || 999999999
var interval = process.argv[5] || 1000
var gray = [100,100,100];

var blessed = require('blessed')
  , contrib = require('blessed-contrib')
  , fs      = require('fs')
  , screen  = blessed.screen()
  , grid = new contrib.grid({rows: 3, cols: 2, screen: screen})
  , gpuutils = grid.set(2, 0, 1, 1, contrib.line,
     { label: 'GPU-Util (%)'
     , style:
       { line: "yellow"
       , text: "green"
       , baseline: gray}
     , xLabelPadding: -10
     , xPadding: -10
     , showLegend: true
     , legend: {width: 10}
     , wholeNumbersOnly: false})
  , gmemories = grid.set(2, 1, 1, 1, contrib.line,
     { label: 'Memory-Usage (MiB)'
     , style:
       { line: "yellow"
       , text: "green"
       , baseline: gray}
     , xLabelPadding: 0
     , xPadding: 0
     , showLegend: true
     , wholeNumbersOnly: false})

function getRecentLogs(n) {
  var logfile = execute("find -mmin -9 -type f -printf '%T@ %p\\0' -maxdepth 3 2> /dev/null | sort -zk 1nr | sed -z 's/^[^ ]* //' | tr '\\0' '\\n' | grep -E '\\./[^.]+\\.log$' | head -n "+n);
  logfile = logfile.replace(/\n$/,'');
  return logfile.split('\n');
}

function makeLine(grid, label, width) {
  var line = grid.set(0, 0, 2, 2, contrib.line, 
     { style:
       { line: "yellow"
       , text: "green"
       , baseline: gray}
     , xLabelPadding: 0
     , xPadding: 0
     , showLegend: true
     , legend: {width: width}
     , wholeNumbersOnly: false //true=do not show fraction in y axis
     , label: label})
  return line
}

var losses = []
function readLines(input, func, line, idx) {
  var remaining = '';  
  var loss = {title:'',x:[],y:[]};
  input.on('data', function(data) {
    remaining += data;
    var index = remaining.indexOf('\n');
    var last  = 0;
    while (index > -1) {
      var l = remaining.substring(last, index);
      last = index + 1;
      var m = func(l);
      var iter = m && m[2];
      var j = m && parseFloat(m[1]);
      if (m !== null && iter >= parseInt(start_iter) && iter <= parseInt(end_iter) && iter % interval == 0) {
        loss.x[loss.x.length] = iter.replace(/000$/,'k');
        loss.y[loss.y.length] = j;
        loss.title = logfile[idx].replace('\.log',': ')+m[1].substr(0,6);
      }
      index = remaining.indexOf('\n', last);
    }
    remaining = remaining.substring(last);
  });
  input.on('end', function() {
    if (remaining.length > 0) {
      var m = func(remaining);
      var iter = m && m[2];
      var j = m && parseFloat(m[1]);
      if (m !== null && iter >= parseInt(start_iter) && iter <= parseInt(end_iter) && iter % interval == 0) {
        loss.x[loss.x.length] = iter.replace(/000$/,'k');
        loss.y[loss.y.length] = j;
        loss.title = logfile[idx].replace('\.log',': ')+m[1].substr(0,6);
      }
    }
    loss.style = {line: colors[idx%4]};
    losses[idx] = loss;
    if (losses.length == logfile.length) {
      for (var i = 0; i < logfile.length; i++) {
        if (!losses[i]) {
          return;
        }
      }
      line.setData(losses);
      screen.render()
    }
  });
  return loss
}


function func(data) {
  //console.log('Line: ' + data);
  var m = data.match(/loss:\s+(.+)\s+.+iter:\s+([0-9]+)\/([0-9]+)/i)
  if (m !== null) {
    return m
  }
  return null
}

var gpuutil = []
var gmemory = []
var colors = ['yellow', 'red', 'cyan', 'white'];

getGPUInfo = function(callback) {
  var csv = execute("nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used --format=csv");
  var obj = csvjson.toObject(csv, {delimiter:','});
  var MAX = 20;
  for (var i = 0; i < obj.length; i++) {
    gpuutil[i] = gpuutil[i] || {title:'', x:[], y:[]};
    gmemory[i] = gmemory[i] || {title:'', x:[], y:[]};
    if (gpuutil[i].x.length >= MAX) { gpuutil[i].x.shift(); gpuutil[i].y.shift(); };
    if (gmemory[i].x.length >= MAX) { gmemory[i].x.shift(); gmemory[i].y.shift(); };
    var percent = parseInt(obj[i]['utilization.gpu [%]'].replace(' %',''));
    var memused = parseInt(obj[i][' memory.used [MiB]'].replace(' MiB',''));
    gpuutil[i].x.push('.');
    gpuutil[i].y.push(percent);
    gpuutil[i].title = i+': '+percent+' %';
    gpuutil[i].style = {line: colors[i%4]};
    gmemory[i].x.push('.');
    gmemory[i].y.push(memused);
    gmemory[i].title = i+': '+memused+' MiB';
    gmemory[i].style = {line: colors[i%4]};
  }
  gpuutils.setData(gpuutil);
  gmemories.setData(gmemory);
};

screen.key(['escape', 'q', 'C-c'], function(ch, key) {
     return process.exit(0);
});
var size_iter = 5000;
screen.key(['+'], function(ch, key) {
     start_iter += size_iter;
});
screen.key(['-','_'], function(ch, key) {
     start_iter = Math.max(0, start_iter-size_iter);
});

var line = makeLine(grid, 'Loss', logfile[1].length+8);

refresh();
screen.render();

function refresh() {
  // Loss Plot
  for (var i = 0; i < logfile.length; i++) {
    var input = fs.createReadStream(logfile[i]);
    readLines(input, func, line, i);
  }
  // GPU Info
  getGPUInfo();
}

setInterval(refresh, 200);
