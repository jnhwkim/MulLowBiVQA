# MISC

Build dashboards using ascii/ansi art and javascript (e.g. loss plot).

### Run

```bash
node dashboard.js logfile.log [start iter] [end iter] [interval]
```

### Installation

For latest nodejs,

```bash
sudo npm cache clean -f
sudo npm install -g n
sudo n stable
```

To install dependencies,

```bash
npm install blessed blessed-contrib csvjson
```
