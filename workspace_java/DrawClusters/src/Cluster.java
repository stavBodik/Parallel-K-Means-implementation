import java.util.ArrayList;

public class Cluster {

	@Override
	public boolean equals(Object obj) {
		return this.id == ((Cluster)obj).getId();
	}

	int id;
	ArrayList<Point> points = new ArrayList<>();
	
	public Cluster(int id) {
		super();
		this.id = id;
	}
	
	
	
	public void addPoint(Point p){
		points.add(p);
	}
	
	public ArrayList<Point> getPoints() {
		return points;
	}
	
	public int getId() {
		return id;
	}



	
	
}
