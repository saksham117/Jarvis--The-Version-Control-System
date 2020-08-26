# Jarvis--The-Version-Control-System

This project is an attempt for a **basic implementation** of a version control system from scratch.


>We have implemented the following commands:
>1. init
>2. status
>3. add
>4. commit
>5. diff
>6. log
>7. hash-object
>8. cat-file
>9. ls-files
>10. reset


These commands are at the core of the VCS, and we have implemented them with the same functionality as in a normal VCS system.
Note: This VCS implementation will run only on a **Unix-based platform.**

**_How to run the VCS:_**

To begin tracking of changes by the VCS, we need to first initialize a repository(directory). This could be done in two ways:
1) telling the VCS to create a new repository
2) giving the VCS path to an existing empty repository. This is done by the init command.

**_For example: python3 jarvis.py init myrepo_**


Thus, first we need to write python3 (version of python) then we need to specify the name of our VCS file, finally the command along with its arguments.

Note: To obtain information about the commands and their arguments type:

**_python3 jarvis.py --help_**


Then change your directory to the current repository. After that, to run the commands you wish you need to specify the complete path of jarvis.py file.
