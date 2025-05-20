from flask import Flask, render_template, request, jsonify
import random
from model_training import select_dream11_team, predict_match_result
app = Flask(__name__)



# Actual player pool with roles and costs (subset for demo)
all_players = [
    # Mumbai Indians
    {"name": "Rohit Sharma", "role": "Batsman", "team": "Mumbai Indians", "cost": 9},
    {"name": "Jasprit Bumrah", "role": "Bowler", "team": "Mumbai Indians", "cost": 8.5},
    {"name": "Hardik Pandya", "role": "All-Rounder", "team": "Mumbai Indians", "cost": 9.5},
    {"name": "Ishan Kishan", "role": "Wicket Keeper", "team": "Mumbai Indians", "cost": 8},
    {"name": "Tilak Varma", "role": "Batsman", "team": "Mumbai Indians", "cost": 7.5},
    {"name": "Suryakumar Yadav", "role": "Batsman", "team": "Mumbai Indians", "cost": 9.4},
    {"name": "Piyush Chawla", "role": "Bowler", "team": "Mumbai Indians", "cost": 8},
    {"name": "Nehal Wadhera", "role": "All-Rounder", "team": "Mumbai Indians", "cost": 7.8},
    {"name": "Gerald Coetzee", "role": "Bowler", "team": "Mumbai Indians", "cost": 8.6},
    {"name": "Tim David", "role": "All-Rounder", "team": "Mumbai Indians", "cost": 8.4},
    {"name": "Arjun Tendulkar", "role": "Bowler", "team": "Mumbai Indians", "cost": 7.5},

    # Chennai Super Kings
    {"name": "MS Dhoni", "role": "Wicket Keeper", "team": "Chennai Super Kings", "cost": 8.5},
    {"name": "Ruturaj Gaikwad", "role": "Batsman", "team": "Chennai Super Kings", "cost": 9},
    {"name": "Ravindra Jadeja", "role": "All-Rounder", "team": "Chennai Super Kings", "cost": 9.2},
    {"name": "Deepak Chahar", "role": "Bowler", "team": "Chennai Super Kings", "cost": 8.4},
    {"name": "Shivam Dube", "role": "All-Rounder", "team": "Chennai Super Kings", "cost": 8.5},
    {"name": "Matheesha Pathirana", "role": "Bowler", "team": "Chennai Super Kings", "cost": 8.7},
    {"name": "Devon Conway", "role": "Batsman", "team": "Chennai Super Kings", "cost": 8.6},
    {"name": "Ben Stokes", "role": "All-Rounder", "team": "Chennai Super Kings", "cost": 9.3},
    {"name": "Moeen Ali", "role": "All-Rounder", "team": "Chennai Super Kings", "cost": 9},
    {"name": "Maheesh Theekshana", "role": "Bowler", "team": "Chennai Super Kings", "cost": 8.2},
    {"name": "Ambati Rayudu", "role": "Batsman", "team": "Chennai Super Kings", "cost": 7.9},

    # Royal Challengers Bangalore
    {"name": "Virat Kohli", "role": "Batsman", "team": "Royal Challengers Bangalore", "cost": 9.8},
    {"name": "Faf du Plessis", "role": "Batsman", "team": "Royal Challengers Bangalore", "cost": 9},
    {"name": "Mohammed Siraj", "role": "Bowler", "team": "Royal Challengers Bangalore", "cost": 8.3},
    {"name": "Dinesh Karthik", "role": "Wicket Keeper", "team": "Royal Challengers Bangalore", "cost": 7.8},
    {"name": "Glenn Maxwell", "role": "All-Rounder", "team": "Royal Challengers Bangalore", "cost": 9.3},
    {"name": "Cameron Green", "role": "All-Rounder", "team": "Royal Challengers Bangalore", "cost": 9},
    {"name": "Karn Sharma", "role": "Bowler", "team": "Royal Challengers Bangalore", "cost": 8},
    {"name": "Anuj Rawat", "role": "Wicket Keeper", "team": "Royal Challengers Bangalore", "cost": 7.5},
    {"name": "Rajat Patidar", "role": "Batsman", "team": "Royal Challengers Bangalore", "cost": 7.7},
    {"name": "Reece Topley", "role": "Bowler", "team": "Royal Challengers Bangalore", "cost": 8.1},
    {"name": "Suyash Prabhudessai", "role": "All-Rounder", "team": "Royal Challengers Bangalore", "cost": 7.4},

    # Gujarat Titans
    {"name": "Shubman Gill", "role": "Batsman", "team": "Gujarat Titans", "cost": 9.3},
    {"name": "Rashid Khan", "role": "Bowler", "team": "Gujarat Titans", "cost": 9},
    {"name": "Wriddhiman Saha", "role": "Wicket Keeper", "team": "Gujarat Titans", "cost": 7.8},
    {"name": "Mohit Sharma", "role": "Bowler", "team": "Gujarat Titans", "cost": 8.4},
    {"name": "Rahul Tewatia", "role": "All-Rounder", "team": "Gujarat Titans", "cost": 8.5},
    {"name": "David Miller", "role": "Batsman", "team": "Gujarat Titans", "cost": 8.7},
    {"name": "Sai Sudharsan", "role": "Batsman", "team": "Gujarat Titans", "cost": 8.3},
    {"name": "Kane Williamson", "role": "Batsman", "team": "Gujarat Titans", "cost": 8.6},
    {"name": "Josh Little", "role": "Bowler", "team": "Gujarat Titans", "cost": 7.9},
    {"name": "Vijay Shankar", "role": "All-Rounder", "team": "Gujarat Titans", "cost": 7.7},
    {"name": "Noor Ahmad", "role": "Bowler", "team": "Gujarat Titans", "cost": 7.8},

 # Kolkata Knight Riders (KKR)
    {"name": "Shreyas Iyer", "role": "Batsman", "team": "Kolkata Knight Riders", "cost": 9},
    {"name": "Andre Russell", "role": "All-Rounder", "team": "Kolkata Knight Riders", "cost": 9.2},
    {"name": "Sunil Narine", "role": "All-Rounder", "team": "Kolkata Knight Riders", "cost": 9},
    {"name": "Rinku Singh", "role": "Batsman", "team": "Kolkata Knight Riders", "cost": 8.4},
    {"name": "Nitish Rana", "role": "Batsman", "team": "Kolkata Knight Riders", "cost": 8.6},
    {"name": "Rahmanullah Gurbaz", "role": "Wicket Keeper", "team": "Kolkata Knight Riders", "cost": 8},
    {"name": "Varun Chakravarthy", "role": "Bowler", "team": "Kolkata Knight Riders", "cost": 8.7},
    {"name": "Vaibhav Arora", "role": "Bowler", "team": "Kolkata Knight Riders", "cost": 7.5},
    {"name": "Mitchell Starc", "role": "Bowler", "team": "Kolkata Knight Riders", "cost": 9.3},
    {"name": "Venkatesh Iyer", "role": "All-Rounder", "team": "Kolkata Knight Riders", "cost": 8.3},
    {"name": "Harshit Rana", "role": "Bowler", "team": "Kolkata Knight Riders", "cost": 7.8},

    # Sunrisers Hyderabad (SRH)
    {"name": "Aiden Markram", "role": "Batsman", "team": "Sunrisers Hyderabad", "cost": 8.8},
    {"name": "Abhishek Sharma", "role": "All-Rounder", "team": "Sunrisers Hyderabad", "cost": 8.4},
    {"name": "Rahul Tripathi", "role": "Batsman", "team": "Sunrisers Hyderabad", "cost": 8.5},
    {"name": "Heinrich Klaasen", "role": "Wicket Keeper", "team": "Sunrisers Hyderabad", "cost": 8.9},
    {"name": "Travis Head", "role": "Batsman", "team": "Sunrisers Hyderabad", "cost": 9.2},
    {"name": "Pat Cummins", "role": "Bowler", "team": "Sunrisers Hyderabad", "cost": 9.5},
    {"name": "T Natarajan", "role": "Bowler", "team": "Sunrisers Hyderabad", "cost": 8},
    {"name": "Marco Jansen", "role": "All-Rounder", "team": "Sunrisers Hyderabad", "cost": 8.4},
    {"name": "Mayank Agarwal", "role": "Batsman", "team": "Sunrisers Hyderabad", "cost": 8.2},
    {"name": "Washington Sundar", "role": "All-Rounder", "team": "Sunrisers Hyderabad", "cost": 8},
    {"name": "Bhuvneshwar Kumar", "role": "Bowler", "team": "Sunrisers Hyderabad", "cost": 8.5},

    # Delhi Capitals (DC)
    {"name": "David Warner", "role": "Batsman", "team": "Delhi Capitals", "cost": 9},
    {"name": "Prithvi Shaw", "role": "Batsman", "team": "Delhi Capitals", "cost": 8.2},
    {"name": "Rishabh Pant", "role": "Wicket Keeper", "team": "Delhi Capitals", "cost": 9},
    {"name": "Axar Patel", "role": "All-Rounder", "team": "Delhi Capitals", "cost": 8.8},
    {"name": "Kuldeep Yadav", "role": "Bowler", "team": "Delhi Capitals", "cost": 8.5},
    {"name": "Anrich Nortje", "role": "Bowler", "team": "Delhi Capitals", "cost": 8.9},
    {"name": "Lalit Yadav", "role": "All-Rounder", "team": "Delhi Capitals", "cost": 7.8},
    {"name": "Mitchell Marsh", "role": "All-Rounder", "team": "Delhi Capitals", "cost": 9.3},
    {"name": "Khaleel Ahmed", "role": "Bowler", "team": "Delhi Capitals", "cost": 8},
    {"name": "Jake Fraser-McGurk", "role": "Batsman", "team": "Delhi Capitals", "cost": 7.6},
    {"name": "Ishant Sharma", "role": "Bowler", "team": "Delhi Capitals", "cost": 7.5},

    # Rajasthan Royals (RR)
    {"name": "Sanju Samson", "role": "Wicket Keeper", "team": "Rajasthan Royals", "cost": 9.1},
    {"name": "Jos Buttler", "role": "Batsman", "team": "Rajasthan Royals", "cost": 9.2},
    {"name": "Yashasvi Jaiswal", "role": "Batsman", "team": "Rajasthan Royals", "cost": 9},
    {"name": "Shimron Hetmyer", "role": "Batsman", "team": "Rajasthan Royals", "cost": 8.3},
    {"name": "Riyan Parag", "role": "All-Rounder", "team": "Rajasthan Royals", "cost": 8.2},
    {"name": "Trent Boult", "role": "Bowler", "team": "Rajasthan Royals", "cost": 8.6},
    {"name": "Ravichandran Ashwin", "role": "All-Rounder", "team": "Rajasthan Royals", "cost": 8.7},
    {"name": "Yuzvendra Chahal", "role": "Bowler", "team": "Rajasthan Royals", "cost": 8.9},
    {"name": "Dhruv Jurel", "role": "Wicket Keeper", "team": "Rajasthan Royals", "cost": 7.8},
    {"name": "Avesh Khan", "role": "Bowler", "team": "Rajasthan Royals", "cost": 8},
    {"name": "Navdeep Saini", "role": "Bowler", "team": "Rajasthan Royals", "cost": 7.7},

    # Punjab Kings (PBKS)
    {"name": "Shikhar Dhawan", "role": "Batsman", "team": "Punjab Kings", "cost": 8.9},
    {"name": "Liam Livingstone", "role": "All-Rounder", "team": "Punjab Kings", "cost": 9},
    {"name": "Jitesh Sharma", "role": "Wicket Keeper", "team": "Punjab Kings", "cost": 8},
    {"name": "Arshdeep Singh", "role": "Bowler", "team": "Punjab Kings", "cost": 8.6},
    {"name": "Sam Curran", "role": "All-Rounder", "team": "Punjab Kings", "cost": 9.3},
    {"name": "Kagiso Rabada", "role": "Bowler", "team": "Punjab Kings", "cost": 9},
    {"name": "Harpreet Brar", "role": "Bowler", "team": "Punjab Kings", "cost": 7.9},
    {"name": "Shahrukh Khan", "role": "Batsman", "team": "Punjab Kings", "cost": 7.7},
    {"name": "Rishi Dhawan", "role": "All-Rounder", "team": "Punjab Kings", "cost": 7.8},
    {"name": "Prabhsimran Singh", "role": "Wicket Keeper", "team": "Punjab Kings", "cost": 8},
    {"name": "Rahul Chahar", "role": "Bowler", "team": "Punjab Kings", "cost": 8},

    # Lucknow Super Giants (LSG)
    {"name": "KL Rahul", "role": "Wicket Keeper", "team": "Lucknow Super Giants", "cost": 9.1},
    {"name": "Quinton de Kock", "role": "Wicket Keeper", "team": "Lucknow Super Giants", "cost": 9},
    {"name": "Marcus Stoinis", "role": "All-Rounder", "team": "Lucknow Super Giants", "cost": 9},
    {"name": "Nicholas Pooran", "role": "Wicket Keeper", "team": "Lucknow Super Giants", "cost": 8.9},
    {"name": "Krunal Pandya", "role": "All-Rounder", "team": "Lucknow Super Giants", "cost": 8.5},
    {"name": "Ravi Bishnoi", "role": "Bowler", "team": "Lucknow Super Giants", "cost": 8.2},
    {"name": "Avesh Khan", "role": "Bowler", "team": "Lucknow Super Giants", "cost": 8},
    {"name": "Deepak Hooda", "role": "All-Rounder", "team": "Lucknow Super Giants", "cost": 8.1},
    {"name": "Ayush Badoni", "role": "Batsman", "team": "Lucknow Super Giants", "cost": 7.9},
    {"name": "Naveen-ul-Haq", "role": "Bowler", "team": "Lucknow Super Giants", "cost": 8},
    {"name": "Mohsin Khan", "role": "Bowler", "team": "Lucknow Super Giants", "cost": 7.6}]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    team1 = data.get("team1")
    team2 = data.get("team2")
    venue = data.get("venue")

    if not team1 or not team2 or not venue:
        return jsonify({"success": False, "error": "Invalid input"})

    # Simulate win probabilities
    team1_win_prob = random.randint(40, 60)
    team2_win_prob = 100 - team1_win_prob

    # Filter players from both selected teams
    team_players = [p for p in all_players if p["team"] in [team1, team2]]

    # Sort by cost (high to low)
    team_players_sorted = sorted(team_players, key=lambda x: x["cost"], reverse=True)

    # Pick best 11
    predicted_team = team_players_sorted[:11]

    # Assign captain and vice-captain from top 11
    if predicted_team:
        captain = predicted_team[0]
        vice_captain = predicted_team[1] if len(predicted_team) > 1 else predicted_team[0]
        for p in predicted_team:
            p["team_role"] = ""
        captain["team_role"] = "Captain"
        vice_captain["team_role"] = "Vice-Captain"

    # Team player count
    team_distribution = {team1: 0, team2: 0}
    for p in predicted_team:
        team_distribution[p["team"]] += 1

    return jsonify({
        "success": True,
        "team": predicted_team,
        "team_distribution": team_distribution,
        "win_prediction": {
            team1: team1_win_prob,
            team2: team2_win_prob
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
