pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.10'
        VENV_DIR = 'venv'
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                
                // Clean workspace
                cleanWs()
                
                // Checkout code
                checkout scm
                
                script {
                    if (isUnix()) {
                        sh '''
                            echo "Creating Python virtual environment..."
                            python3 -m venv ${VENV_DIR}
                            . ${VENV_DIR}/bin/activate
                            
                            echo "Upgrading pip..."
                            pip install --upgrade pip
                            
                            echo "Installing dependencies..."
                            pip install -r requirements.txt
                            
                            echo "Verifying installation..."
                            pip list
                        '''
                    } else {
                        bat '''
                            echo "Creating Python virtual environment..."
                            python -m venv %VENV_DIR%
                            
                            echo "Activating virtual environment..."
                            call %VENV_DIR%\\Scripts\\activate.bat
                            
                            echo "Upgrading pip..."
                            python -m pip install --upgrade pip
                            
                            echo "Installing dependencies..."
                            pip install -r requirements.txt
                            
                            echo "Verifying installation..."
                            pip list
                        '''
                    }
                }
                
                echo 'Environment setup completed successfully!'
            }
        }
        
        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . ${VENV_DIR}/bin/activate
                            
                            echo "Compiling pipeline..."
                            python pipeline.py
                            
                            echo "Verifying pipeline.yaml exists..."
                            if [ -f "pipeline.yaml" ]; then
                                echo "✓ pipeline.yaml generated successfully"
                                ls -lh pipeline.yaml
                            else
                                echo "✗ pipeline.yaml not found"
                                exit 1
                            fi
                        '''
                    } else {
                        bat '''
                            call %VENV_DIR%\\Scripts\\activate.bat
                            
                            echo "Compiling pipeline..."
                            python pipeline.py
                            
                            echo "Verifying pipeline.yaml exists..."
                            if exist pipeline.yaml (
                                echo ✓ pipeline.yaml generated successfully
                                dir pipeline.yaml
                            ) else (
                                echo ✗ pipeline.yaml not found
                                exit /b 1
                            )
                        '''
                    }
                }
                
                echo 'Pipeline compilation completed successfully!'
            }
        }
        
        stage('Code Quality Check') {
            steps {
                echo 'Running code quality checks...'
                
                script {
                    if (isUnix()) {
                        sh '''
                            . ${VENV_DIR}/bin/activate
                            
                            echo "Checking Python syntax..."
                            python -m py_compile src/pipeline_components.py
                            python -m py_compile src/model_training.py
                            python -m py_compile pipeline.py
                            
                            echo "All Python files are syntactically correct!"
                        '''
                    } else {
                        bat '''
                            call %VENV_DIR%\\Scripts\\activate.bat
                            
                            echo "Checking Python syntax..."
                            python -m py_compile src\\pipeline_components.py
                            python -m py_compile src\\model_training.py
                            python -m py_compile pipeline.py
                            
                            echo "All Python files are syntactically correct!"
                        '''
                    }
                }
                
                echo 'Code quality check completed successfully!'
            }
        }
    }
    
    post {
        success {
            echo '========================================='
            echo 'Pipeline execution completed successfully!'
            echo '========================================='
            echo 'All stages passed:'
            echo '  ✓ Environment Setup'
            echo '  ✓ Pipeline Compilation'
            echo '  ✓ Code Quality Check'
            echo '========================================='
            
            // Archive the compiled pipeline
            archiveArtifacts artifacts: 'pipeline.yaml', fingerprint: true
        }
        
        failure {
            echo '========================================='
            echo 'Pipeline execution failed!'
            echo 'Please check the logs above for details.'
            echo '========================================='
        }
        
        always {
            echo 'Cleaning up...'
            // Clean up workspace if needed
        }
    }
}
