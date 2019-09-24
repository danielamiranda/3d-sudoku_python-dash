# 3D-Sudoku prototype with Python-Dash

Sudoku is a popular game originated from the work of the 17th century Swiss mathematician, Leonard Euler. He suggested the idea of ‘Latin squares’, which consist of grids of equal dimensions in which every symbol occurs exactly once in every row and every column.
This project presents an optimization model that generates and solves 3D Sudoku puzzles. This model is embedded into Graphical User Interface (GUI) in Python-Dash that allows users to play 3D Sudoku

## Getting Started

To run this code, follow these [instructions](https://m.wikihow.com/Use-Windows-Command-Prompt-to-Run-a-Python-File)

The GUI will be displayed in your browswer in this address: http://127.0.0.1:8050/


### Prerequisites

The list of packages you need to run this code are listed in the [requirements.txt](requirements.txt) file.
The Python version used to develop this code is 3.7.3


### Playing

To play select the level of difficulty and press the play button. Once you click on the play button, a puzzle is randomly selected and displayed on the right hand side. 
This puzzle is represented in a 3D cube with balls of five different colors: Blue, green, red, yellow, and grey.
Your goal is to fill the grey balls with the other colors following this simple rule: there must be ONLY ONE ball of each color per row and per column. If there is a repeated color, you lose!

***NOTE:*** Sometimes the puzzle is not displayed automatically. Then an additional click on the ‘3D-Sudoku’ area is needed to get the puzzle and timer 
***NOTE 2:*** The cube position resets every time you click on a new ball. This is an area for improvement


## Deployment

This puzzle is deployed online in [3D-sudoku heroku](https://jads-3dsudoku.herokuapp.com/)
The username and password can be accessed upon request


## Authors

* **Aynaz Kafashan** 
* **Daniela Miranda**
* **Juhi Chandra** 
* **Pallabi Sengupta**  


## License

This project is licensed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License) 

## Acknowledgments

* Raul Beeldsnijder: for presenting this nice challenge to us
* Dick den Hertog: for helping us developing our abilities in solving problems using optimization 
