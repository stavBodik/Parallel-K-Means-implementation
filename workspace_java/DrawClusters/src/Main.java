
 
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

import com.sun.java_cup.internal.runtime.Symbol;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.ArcType;
import javafx.stage.Stage;
 
public class Main extends Application {
 
	private static int i;
    public static void main(String[] args) {
    	
    	launch(args);
    }
 
    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Drawing Operations Test");
        Group root = new Group();
        Canvas canvas = new Canvas(800,800);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        
        
        ArrayList<Cluster> clusters = new ArrayList<>();
        

        loadClusters(clusters);
        
        System.out.println(clusters.size());

        drawClusters(gc,clusters,0.1);
        root.getChildren().add(canvas);
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

    private void loadClusters(ArrayList<Cluster> clusters){
    	
    	 //Name of the file
        String fileName="C://PK//Stav_Java_App.txt";
      //  String fileName="C://PK//clusters_best.txt";
        try{

           //Create object of FileReader
           FileReader inputFile = new FileReader(fileName);

           //Instantiate the BufferedReader Class
           BufferedReader bufferReader = new BufferedReader(inputFile);

           //Variable to hold the one line data
           String line;

           // Read file line by line and print on the console
           int clusterID;
           int koko=0;
           while ((line = bufferReader.readLine()) != null)   {
        	  // System.out.println("LINE " + koko +" : " +Arrays.toString(line.split(" ")) );
        	   clusterID = Integer.parseInt(line.split(" ")[0]);
        	   
        	   Cluster newCluster = new Cluster(clusterID);
        	   
        	   if(!clusters.contains(newCluster)){
        		   clusters.add(newCluster);   
        	   }
        	   
        	   Point p = new Point(Double.parseDouble(line.split(" ")[1]), Double.parseDouble(line.split(" ")[2]));
        	   for(int i=0; i<clusters.size(); i++){
        		   if(clusters.get(i).getId()==clusterID){
        			   clusters.get(i).addPoint(p);
        		   }
        	   }
        	   
        	   koko++;
           }
           
           //Close the buffer reader
           bufferReader.close();
        }catch(Exception e){
           System.out.println("Error while reading file line by line:" + e.getMessage());                      
        }
    	
        System.out.println("N CLUSTERS " + clusters.size());
        for(int i=0; i<clusters.size(); i++){
        	System.out.println("cluster id " + clusters.get(i).getId()+ " n points : "+clusters.get(i).getPoints().size() );
        }
        
    }
    
    private void drawClusters(GraphicsContext gc,ArrayList<Cluster> clusters,double zoom) {
    	 
    	for(int i=0; i<clusters.size(); i++){
    		Paint color = null;
    		switch (clusters.get(i).getId()) {
			case 0:
				color = Color.GREEN;
				break;
				
			case 1:
				color = Color.ORANGE;
				break;
				
				
			case 2:
				color = Color.BLUE;
				break;
			}
    		
    		
    		
    		for(int j=0; j<clusters.get(i).getPoints().size(); j++){
    			gc.setFill(color);
            	
    			double drawX= (clusters.get(i).getPoints().get(j).getX()*zoom);
    			double drawY= (clusters.get(i).getPoints().get(j).getY()*zoom);
    			
    			gc.fillOval(drawX,drawY,1,1);
            	
            	//System.out.println("Drawing " +drawX*8 +","+drawY*8);
    		}
    		
    		
    	}
    	
    	double centers[] = {259.209021,187.531293,646.560444,403.558530,256.777594,617.149346};
    	
    	
    	gc.setFill(Color.RED);
    	gc.fillOval(centers[0]*zoom,centers[1]*zoom,4,4);
    	
    	gc.setFill(Color.RED);
    	gc.fillOval(centers[2]*zoom,centers[3]*zoom,4,4);
    	
    	gc.setFill(Color.RED);
    	gc.fillOval(centers[4]*zoom,centers[5]*zoom,4,4);

    	
    	
    }
}