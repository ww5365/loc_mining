package driver.loc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;


import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import weka.classifiers.Classifier;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class UserClassifier {


	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		
		JobConf job = new JobConf();
		job.setJarByClass(UserClassifier.class);
		
		job.setMapOutputKeyClass(Text.class);//设定map输出的key使用文本类，这样格式
		job.setMapOutputValueClass(Text.class);
		
		job.setInputFormat(TextInputFormat.class);//一行行的输入数据		
		job.setOutputFormat(TextOutputFormat.class);
		
		job.setMapperClass(MapUserCluster.class);
		job.setReducerClass(RedUserStat.class);
		
		job.set("mapred.job.priority", "VERY_HIGH");
		job.set("mapred.job.queue.name", "map-international");		
		
		job.setNumReduceTasks(400);
		
		//设置hadoop平台上的输入文件,匹配，多输入
		
		//FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/user_feature_test");
		
		FileInputFormat.setInputPaths(job,"/app/map/map-international/wangwei/tmp/user_feature");
		
		//  /app/lbs/lbs-guiji/ident_traj/mark_i2c/201508[0-2][0-9]
		
		Path outPath = new Path("/app/map/map-international/sfc/datbases/allmap_loc");
		FileSystem fs = outPath.getFileSystem(job);
		
		if(fs.exists(outPath)){
			fs.delete(outPath,true); //输出文件存在的情况下，删除此文件夹
		}
		
		FileOutputFormat.setOutputPath(job, outPath);
		
		//设置使用hadoop平台，临时文件
		
		job.set("mapred.create.symlink", "yes");
		job.set("mapred.cache.files", "/app/map/map-international/wangwei/tmp/model/j48.model#j48.model");

		
		//解析第三方包的参数		
		String[] otherArgs = new GenericOptionsParser(job, args).getRemainingArgs();	
		
		System.out.println("job created and args: "+otherArgs.length);
		
		DistributedCache.createSymlink(job);
		
		int n = JobClient.runJobReturnExitCode(job);
		if(n != 0 ){
			System.out.println("Job FAIL");
		}
		System.exit(0);
		
	}
	
	 
    
	static class MapUserCluster extends MapReduceBase  implements Mapper<LongWritable, Text, Text, Text> {
		
		public String epsion;
		public String minPoint;		
		public ArrayList<String> userPoiList;
		public String lastCuid;
		public boolean firstLineFlag;
		
		
		/*
		 * configure 完成mapper的初始化工作(non-Javadoc)
		 * 
		 */
		public void configure(JobConf context)  
		{
			try{   		
	    		System.out.println("Mapper configure success!!!");
			}
			catch(Exception e){
				e.printStackTrace();
				System.err.println("Mapper configure error!!");
			}
		}
		
		
		/*
		 * @see org.apache.hadoop.mapred.Mapper#map(java.lang.Object, java.lang.Object, org.apache.hadoop.mapred.OutputCollector, org.apache.hadoop.mapred.Reporter)
		 * @Override
		 */
		
		public void map(LongWritable key, Text line,OutputCollector<Text, Text> output, Reporter rep)throws IOException {
			

			String word = line.toString();			
			//System.out.println("map:line "+word);			
			String[] parts=word.split("\t");
			
			
			//用户特征数据，有18个字段
			//<cuid,clusterID,Centerx,centery,wifi_entropy,ratio_present_day,avg_stay_time,
			// ratio_daytime,ratio_night,ratio_weekend,ratio_slot_0,ratio_slot_1,ratio_slot_2,ratio_slot_3,
			//ratio_slot_4,ratio_slot_5,ratio_slot_6,ratio_slot_7,>

			if(parts.length!=18){
				return;				
			}
			
			
			String cuidStr = parts[0].trim();
			String clusterID = parts[1].trim();

			
			if(!("-1".equalsIgnoreCase(clusterID))){				
				output.collect(new Text(cuidStr),line);		//直接输出到reduce中
				
				//System.out.println("line:"+word);
			}  
		}
		
		
	
		
	}
	
	
	/*
	 * @reducer
	 */
	
	static class RedUserStat extends MapReduceBase implements Reducer<Text, Text, Text, Text>{
		
		
		/*
		 * configure 完成reducer的初始化工作(non-Javadoc)
		 * 
		 */
		public void configure(JobConf context)  
		{
			try{	
			
				//读取序列化数据：反序列化
				Path path[]=null;			
				path=DistributedCache.getLocalCacheFiles(context);				
				
	    		System.out.println("reducer configure success!!!");
			}
			catch(Exception e){
				e.printStackTrace();
				System.err.println("reducer configure error!!");
			}
		}
	

		
		/*
		 * 构造特征值的Instance
		 */
		
		public  Instance makeInstance(String feature,Instances testSet){
			
			if(feature==null){
				return null;
			}
			
			//输入:(cuid，clusterID，x，y，14个特征字段)
			
			String[] parts = feature.split("\t");
			
			if(parts.length!=18){
				
				System.out.println("parts len: "+ parts.length);
				return null;
			}
			
			Instance testInst = new Instance(15);
			testInst.setDataset(testSet);
			
			for(int i=4;i<parts.length;i++){
				testInst.setValue(i-4, Double.parseDouble(parts[i]));
			}
			
			return testInst;			
			
		}
		
		
		

		@Override
		public void reduce(Text key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter rep)
				throws IOException {
			
			//接收数据：<cuid，(cuid，clusterID，x，y，14个特征字段),....>
			
			
			double lastHome = 0;
			double lastCom = 0;
			String homeRes = null;
			String comRes = null;
			String homeClusterIDRes = null;
			String comClusterIDRes = null;
			
			
			
			while(values.hasNext()){
				
				
				String outstrstr = values.next().toString();
				
				String[] parts = outstrstr.split("\t");
				String cuid = parts[0].trim();
				String clusterID = parts[1].trim();	
				String centerX = parts[2].trim();
				String centerY = parts[3].trim();
				
				/*
				 * 读取序列化的：模型文件：j48
				 * 
				 * @relation driverModel   
				 * @attribute wifi_entropy numeric
				 * @attribute present_day numeric
				 * @attribute avg_stay_time numeric
				 * @attribute day_time numeric
				 * @attribute night_time numeric
				 * @attribute weekend numeric
				 * @attribute slot_time_0 numeric
				 * @attribute slot_time_1 numeric
				 * @attribute slot_time_2 numeric
				 * @attribute slot_time_3 numeric
				 * @attribute slot_time_4 numeric
				 * @attribute slot_time_5 numeric
				 * @attribute slot_time_6 numeric
				 * @attribute slot_time_7 numeric
				 * @attribute label {H,C,O}
				 */
				
				Object obj[]= null;			
				try {
					obj = SerializationHelper.readAll("j48.model");
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				Classifier tree = (Classifier)obj[0];
				Instances trainSet = (Instances) obj[1];
				
				Instances testSet = trainSet.stringFreeStructure();  //最后一列，不需要设置值，但属性要有			

				Instance testInst = makeInstance(outstrstr,testSet);
				
				//System.out.println("out:"+outstrstr);
				//System.out.println("inst:"+testInst);
				
				
				if(!trainSet.equalHeaders(testSet)){
					System.out.println("train and test set header not compatible!");
					continue;
				}
				
				
				double pred = -1;
				double[] dist = null;
				String preClass = null;

				
				try {
					pred = tree.classifyInstance(testInst); // 将测试集中的实例预测什么类别，索引值
					dist = tree.distributionForInstance(testInst);
					preClass = testSet.classAttribute().value((int) pred);

					//System.out.println("pre:preClass " + pred + " " + preClass);

					//System.out.println(Utils.arrayToString(dist));


					if (pred == 0 && dist[(int) pred] > lastHome) {// 预测为家且置信度最大的记录
						homeRes = new String(centerX + "\t" + centerY + "\t" + preClass);
						lastHome = dist[(int) pred];
						
						homeClusterIDRes = clusterID;

					}

					if (pred == 1 && dist[(int) pred] > lastCom) {// 预测为公司且置信度最大的记录
						comRes = new String(centerX + "\t" + centerY + "\t" + preClass);
						lastCom = dist[(int) pred];						
						comClusterIDRes = clusterID;
					}
					
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					continue;
				}		
			
			}
			
			//每个用户只有一个家或公司的结果
			if (!("-1".equals(homeClusterIDRes)) && homeRes != null) {
				output.collect(key, new Text(homeClusterIDRes + "\t"+ homeRes));
				}

			if (!("-1".equals(comClusterIDRes)) && comRes != null) {
				output.collect(key, new Text(comClusterIDRes + "\t"+ comRes));
				}	

		}//end reduce
		
	}//end RedUserStat
}
