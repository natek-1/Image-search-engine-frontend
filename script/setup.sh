sudo apt-get update
sudo apt-get upgrade -y


# installing docker
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce -y
sudo systemctl status docker

sudo usermod -aG docker ubuntu
newgrp docker

## Aws cli installation
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install #got /usr/local/bin/aws --version # You can now run: /usr/local/bin/aws --version

## Github Runner configuration (dependant on the project so check the code from your direct repo)
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.316.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
echo "64a47e18119f0c5d70e21b6050472c2af3f582633c9678d40cb5bcb852bcc18f  actions-runner-linux-x64-2.316.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.316.0.tar.gz

./config.sh --url https://github.com/natek-1/Image-serch-engine-data-collection --token <personal-token>
./run.sh #to run self-hosted runner cd actions-runner  and ./run.sh

## Add Github runner as a service
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status

## To stop the service
sudo ./svc.sh stop
sudo ./svc.sh uninstall