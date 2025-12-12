### How to run the integrated tests:

#### 1) Download and Install TornadoVM locally:

*Linux (x86_64)*

```bash
wget https://github.com/beehive-lab/TornadoVM/releases/download/v2.1.0/tornadovm-2.1.0-opencl-linux-amd64.zip
unzip tornadovm-2.1.0-opencl-linux-amd64.zip
# Replace <path-to-sdk> manually with the absolute path of the extracted folder
export TORNADO_SDK="<path-to-sdk>/tornadovm-2.1.0-opencl"
export PATH=$TORNADO_SDK/bin:$PATH

tornado --devices
tornado --version
```

*macOS (Apple Silicon)*

```bash
wget https://github.com/beehive-lab/TornadoVM/releases/download/v2.1.0/tornadovm-2.1.0-opencl-mac-aarch64.zip
unzip tornadovm-2.1.0-opencl-mac-aarch64.zip
# Replace <path-to-sdk> manually with the absolute path of the extracted folder
export TORNADO_SDK="<path-to-sdk>/tornadovm-2.1.0-opencl"
export PATH=$TORNADO_SDK/bin:$PATH

tornado --devices
tornado --version
```

Note that the above steps:
- Set the `TORNADOVM_SDK` environment variable to the TornadoVM SDK path.
- `TORNADO_SDK` contains the `tornado-argfile` with all the JVM arguments required to enable TornadoVM.
- ⚠️ The `tornado-argfile` should be used for *building* and *running* the Quarkus application (see section Building & Running the Quarkus Application).

#### 2) Build Quarkus-langchain4j:

```bash
cd ~
git clone git@github.com:quarkiverse/quarkus-langchain4j.git
cd ~/quarkus-langchain4j
mvn clean install -DskipTests
```

#### 3) Run the integrated tests:

##### 3.1 Deploy the Quarkus app:

```bash
cd ~/quarkus-langchain4j/integration-tests/gpullama3
```
- For *dev* mode, run:
```
mvn quarkus:dev
```

- For *production* mode, run:
```bash
java @$TORNADO_SDK/tornado-argfile -jar target/quarkus-app/quarkus-run.jar
```
##### 3.2 Send requests to the Quarkus app:

when quarkus is running, open a new terminal and run:

```bash
curl http://localhost:8080/chat/blocking
curl http://localhost:8080/chat/streaming
```
