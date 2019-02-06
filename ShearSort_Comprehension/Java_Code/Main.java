import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class Main {

	private static 	Random rand = new Random();
	
	public static void main(String[] args) throws InterruptedException{
		
		int matrixSize = 36;
		int matrix[][] = generateRandomMatrixNN(matrixSize);

		final long startTime = System.currentTimeMillis();

		for(int i=0; i<matrix.length+1; i++){
			if(i%2==0){
				matrix = shearSortMatrixHorizontal(matrix);
			}else{
				matrix = shearSortMatrixVertical(matrix);
			}
		}

		final long endTime = System.currentTimeMillis();
		System.out.println((endTime-startTime)+"\n");
		printMatrix(matrix);
	}
	
	private static int[][] generateRandomMatrixNN(int sizeN){
		int matrix[][] = new int[sizeN][sizeN];
		for(int i=0; i<sizeN; i++){
			for(int j=0; j<sizeN; j++){
				matrix[i][j]= randInt(0,100);
			}
		}
		return matrix;
	}
	
	public static int randInt(int min, int max) {
	    int randomNum = rand.nextInt((max - min) + 1) + min;
	    return randomNum;
	}
	
	private static int[][] shearSortMatrixVertical(int matrix[][]) throws InterruptedException{
		
		int nProcessors = Runtime.getRuntime().availableProcessors();
	    ExecutorService executorService = Executors.newFixedThreadPool(nProcessors); 
	    
		for(int i=0; i<matrix.length; i++){
			int ii=i;
			executorService.submit(new Runnable() {
				@Override
				public void run() {
					try {
						int matrixCol[] = getMatrixColAsArray(matrix,ii);
						matrixCol = oddEvenSort(matrixCol, true);
						setMatrixColFromArray(matrix, matrixCol, ii);} 
					catch (InterruptedException e) {e.printStackTrace();}
				}
			});
		}
		
		executorService.shutdown();
		executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		
		return matrix;
	}
	
	private static int[] getMatrixColAsArray(int[][] matrix,int colIndex){
		int matrixCol[] = new int[matrix.length];
		for(int i=0; i<matrix.length; i++){
			matrixCol[i] = matrix[i][colIndex];
		}
		return matrixCol;
	}
	
	private static int[][] setMatrixColFromArray(int[][] matrix,int matrixCol[],int colIndex){
		for(int i=0; i<matrix.length; i++){
			matrix[i][colIndex] = matrixCol[i];
		}
		return matrix;
	}
	
	private static int[][] shearSortMatrixHorizontal(int [][] matrix) throws InterruptedException{
		
	    int nProcessors = Runtime.getRuntime().availableProcessors();
	    ExecutorService executorService = Executors.newFixedThreadPool(nProcessors); 
	    
		for(int i=0; i<matrix.length; i++){
			int ii=i;
			if(i%2==0){
				executorService.submit(new Runnable() {
					@Override
					public void run() {
						try {
							matrix[ii] = oddEvenSort(matrix[ii],true);
						} catch (InterruptedException e) {e.printStackTrace();}
					}
				});
			}else{
				executorService.submit(new Runnable() {
					@Override
					public void run() {
						try {
							matrix[ii] = oddEvenSort(matrix[ii],false);
						} catch (InterruptedException e) {e.printStackTrace();}
					}
				});
			}
		}
		
		executorService.shutdown();
		executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

		
		return matrix;
	}
	
	private static void printMatrix(int[][] matrix){
		for(int i=0; i<matrix.length; i++){
			
			System.out.println(Arrays.toString(matrix[i]));
		}
	}
	
	public static int[] oddEvenSort(int []a,boolean isLeftToRight) throws InterruptedException{
		
		int nIterations=a.length;
	    int nProcessors = Runtime.getRuntime().availableProcessors();
		
		for(int i=0; i<nIterations; i++){
		    ExecutorService executorService = Executors.newFixedThreadPool(nProcessors); 

			if(i%2==0){
				for(int j=0; j<a.length; j+=2){
					int jf=j;
					executorService.submit(new Runnable() {
						@Override
						public void run() {
							if(isLeftToRight){
								if(a[jf]>a[jf+1]){
									swap(jf,jf+1,a);
								}
							}else{
								if(a[jf]<a[jf+1]){
									swap(jf,jf+1,a);
								}
							}
						}
					});
				}
			}else{
				for(int j=1; j<a.length; j+=2){
					int jf=j;
					executorService.submit(new Runnable() {
						@Override
						public void run() {
							if(isLeftToRight){
								if(jf!=(a.length-1) && a[jf]>a[jf+1]){
									swap(jf,jf+1,a);
								}
							}else{
								if(jf!=(a.length-1) && a[jf]<a[jf+1]){
									swap(jf,jf+1,a);
								}
							}
						}
					});	
				}
			}
			
			executorService.shutdown();
			executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		}
		return a;
	}
	
	private static void swap(int leftIndex,int rightIndex,int[] a){
		int temp=a[leftIndex];
		a[leftIndex]=a[rightIndex];
		a[rightIndex]=temp;
	}
}
