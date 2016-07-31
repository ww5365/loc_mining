package driver.loc;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;




public class FeatureStat {
	
	private static HashMap<String, ArrayList<String>> userPoiGroupByCluster = new HashMap<String, ArrayList<String>>();
	

	/*
	 * @return ĳ���û������մغ����ά�ȣ����з�����
	 * 
	 * ѵ��ģ�ͼ��ϣ�
	 * <clusterID1,(x,y,time,Wifi,clusterID,label?),(x,y,time,Wifi,clusterID,label?),...>
	 * 
	 * Ԥ�⼯�ϣ�
	 * <clusterID1,(x,y,time,Wifi,clusterID),(x,y,time,Wifi,clusterID),...>
	 */
	public static int init(ArrayList<String> userPoiList){
		
		//�����һ���û�������
		userPoiGroupByCluster.clear();
		
		//���մغţ��Դ��û������ݽ��з���
		
		if(userPoiList.size()<=0){
			return 0;
		}
		
		int clusterNum=0;
		String word;
		String[] parts;		
		
		for(int i=0;i<userPoiList.size();i++){
			word = userPoiList.get(i).trim();
			parts = word.split("\t");			
			String clusterID = parts[4].trim();			
			if(!userPoiGroupByCluster.containsKey(clusterID)){
				userPoiGroupByCluster.put(clusterID, new ArrayList<String>());	
				clusterNum++;
			}
			
			userPoiGroupByCluster.get(clusterID).add(word);	
		}
		
		return clusterNum;
	}
	
	
	/*
	 * @desc ͳ���û���ÿ���ص����ĵ�
	 */
	
	public static void statCenterCordinate(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userTmp = null;
		String clusterIDTmp = null;
		String[] parts;		
		
		for (Entry<String, ArrayList<String>> en : enties) {
			
			double maxLogitude = 0;
			double minLogitude = Double.MAX_VALUE;		
			double maxLatitude = 0;		
			double minLatitude = Double.MAX_VALUE;
			double x;
			double y;
			
			clusterIDTmp = en.getKey();
			userTmp = en.getValue();
			for (int i = 0; i < userTmp.size(); i++) {
				
				//System.out.println(clusterIDTmp + "\t" + userTmp.get(i));
				parts = userTmp.get(i).split("\t");
				
				x = Double.parseDouble(parts[0]);//x ����
				y = Double.parseDouble(parts[1]); //y ����
				
				if(x>maxLogitude){
					maxLogitude = x;
				}
				
				if(x<minLogitude){
					minLogitude = x;
				}
				
				if(y>maxLatitude){
					maxLatitude = y;
				}
				
				if(y<minLatitude){
					minLatitude = y;
				}				
			}
			
			BigDecimal xx = new BigDecimal((maxLogitude+minLogitude)/2);
			BigDecimal yy = new BigDecimal((maxLatitude+minLatitude)/2);
			
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//�����û����ڴص����ĵ㣬���뵽����list��
			statRes.get(clusterIDTmp).add(xx.toString());
			statRes.get(clusterIDTmp).add(yy.toString());			
			
		}		
		
		
	}
	
	

	/*
	 * @desc  ͳ���û�ÿ���ص�wifi������
	 * 
	 * ���㷽����
	 * 
	 * ÿ�����г��ֵ�wifi��������ȥ�أ���Ϊy��
	 * ÿ������ÿ��wifi�ĳ��ִ�����Ϊ<x1,x2,x3...>��
	 * 
	 * �õ�ÿ����wifi�ĸ��ʷֲ���<x1/y,x2/y,....>
	 * 
	 * ʹ������ķֲ�������ôص��أ�
	 * 
	 */
	
	public static void statWifiEntropy(HashMap<String, ArrayList<String>> statRes){		
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	
		HashMap<String, Long> eachWifiCount = new HashMap<String, Long>();		
		ArrayList<Double> wifiRatio = new ArrayList<Double>();
		
		for (Entry<String, ArrayList<String>> en : enties) {
			
			wifiRatio.clear();
			eachWifiCount.clear();
			long wifiTotal = 0;			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//�غ�ά�ȵ�����wifi����ͳ��
				parts = userPoiTmp.get(i).split("\t");
				wifiTotal++;
				String tmpWifi = parts[3].trim();
				if(!eachWifiCount.containsKey(tmpWifi)){
					eachWifiCount.put(tmpWifi, new Long(1));
				}else{
					Long tmpCount = eachWifiCount.get(tmpWifi);
					eachWifiCount.put(tmpWifi, ++tmpCount);
				}
			}
			
			//��ͬ���У���ͬ��wifiֵ��ռ�ĸ���
			if(wifiTotal>0){
				Set<Entry<String, Long>> tmpEnties = eachWifiCount.entrySet();				
				for(Entry<String, Long> entmp:tmpEnties){
					Double ratio = new Double(entmp.getValue().doubleValue()/wifiTotal);
					wifiRatio.add(ratio);					
				}				
			}
			
			//System.out.println(clusterIDTmp+"\t"+wifiRatio);
			
			double entropy = UtilForMe.CalEntropy(wifiRatio);
			
			//System.out.println(clusterIDTmp+"\t"+entropy);
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//�����û����ڴص����ĵ㣬���뵽����list��
			statRes.get(clusterIDTmp).add(Double.toString(entropy));
		}		
	}
	
	

	

	/*
	 * @ͳ��ÿ���û����ڲ�ͬ�Ĵ��г��ֵ����� ���� ��ȥ�������ܹ���λ������
	 * ���㷽����
	 * ���û������д��У����ظ��ĳ��ֵ����ӣ���Ϊx��
	 * ĳ�����У����ظ��ĳ��ֵ����ӣ���Ϊy��
	 * 
	 * ����� ÿ���أ�y/x
	 * 
	 */
	public static void statRatioPresentDay(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		HashSet<String> allDate = new HashSet<String>();
		HashMap<String,HashSet<String>> oneClusterPresentDay = new HashMap<String, HashSet<String>>();		
		oneClusterPresentDay.clear();
		allDate.clear();
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//�غ�ά��
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpDate = UtilForMe.timeStampToDate(tmpTimeStamp, "yyyy-MM-dd");	
				
				allDate.add(tmpDate);//��¼���д��еĲ��ظ���ʱ��
				
				if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
					oneClusterPresentDay.put(clusterIDTmp, new HashSet<String>());
				}				
				oneClusterPresentDay.get(clusterIDTmp).add(tmpDate);//��¼ÿ�����У����ظ���ʱ��				
			}			
		}
		
		
		int allPresentDay = allDate.size();
		
		
		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			int oneCluPresentDay = oneClusterPresentDay.get(clusterIDTmp).size();
			
			double ratioPresentDay = 0;
			
			if(allPresentDay!=0){
				ratioPresentDay = (double)oneCluPresentDay/allPresentDay;
			}
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//�����û����ڴص����ĵ㣬���뵽����list��
			statRes.get(clusterIDTmp).add(Double.toString(ratioPresentDay));
			
			//System.out.println("clusterIDTmp:ratio: "+ clusterIDTmp +"\t"+ratioPresentDay);
			
		}		
		
	}
	
	/*
	 * @ͳ��ÿ���û����ڲ�ͬ�Ĵ� ,  ÿ���ƽ��ͣ��ʱ��
	 * 
	 * ���㷽����
	 * �ض����У����ֵ�������ȥ�أ� ��Ϊy��
	 * ���ֵ���y���У�ÿ��ͣ��ʱ��,�����ܺ���Ϊx��
	 * 
	 * �����x/y
	 * 
	 */
	public static void statAvgStayTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		HashMap<String,Long> oneClusterMinStayTime = new HashMap<String, Long>();	
		HashMap<String,Long> oneClusterMaxStayTime = new HashMap<String, Long>();	

		
		for (Entry<String, ArrayList<String>> en : enties) {	
			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			oneClusterMinStayTime.clear();
			oneClusterMaxStayTime.clear();
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpDate = UtilForMe.timeStampToDate(tmpTimeStamp, "yyyy-MM-dd");	
				
				if(!oneClusterMinStayTime.containsKey(tmpDate)){
					oneClusterMinStayTime.put(tmpDate, new Long(Long.MAX_VALUE));
				}
				
				if(!oneClusterMaxStayTime.containsKey(tmpDate)){
					oneClusterMaxStayTime.put(tmpDate, new Long(0));
				}
				
				Long tmpTimeStampLong = new Long(tmpTimeStamp); 
				
				//Ѱ��ĳ������ʱ�����Сʱ��
				if(tmpTimeStampLong<oneClusterMinStayTime.get(tmpDate)){
					oneClusterMinStayTime.put(tmpDate, tmpTimeStampLong);
				}
				
				if(tmpTimeStampLong>oneClusterMaxStayTime.get(tmpDate)){
					oneClusterMaxStayTime.put(tmpDate, tmpTimeStampLong);
				}					
			}
			
			long totalTime = 0;
			long totalDay = 0;
			Set<Entry<String, Long>> minEntries = oneClusterMinStayTime.entrySet();
			
			for(Entry<String, Long> minEn:minEntries){
				String tmpDateKey = minEn.getKey();
				Long minTime = minEn.getValue();				
				Long maxTimeLong = oneClusterMaxStayTime.get(tmpDateKey);
				
				totalDay++;
				if(maxTimeLong-minTime>=0){
					totalTime+=(maxTimeLong.longValue()-minTime.longValue());
				}				
			}
			
			long avgStayTime = 0;  //ÿ��ƽ��ͣ��ʱ�� ��λs
			if(totalDay!=0){
				avgStayTime = totalTime/totalDay;
			}
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//�����û����ڴص����ĵ㣬���뵽����list��
			statRes.get(clusterIDTmp).add(Long.toString(avgStayTime));
			
			//System.out.println("clusterIDTmp:avgStayTime:" + clusterIDTmp + "\t" + avgStayTime);			
			
		}		
	}
	
	/*
	 * @ͳ��ÿ���û�����ͬ�� ��������ӱ��� �û�ȫ�����춨λ��������
	 * 
	 * ���㷽����
	 * �û����춨λ��������ȥ�أ� ��Ϊy��
	 * ĳ�����У������������Ϊx��
	 * 
	 * ���죺6:00~18:00
	 * 
	 * �����x/y
	 * 
	 */
	public static void statRatioDayTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		String dayTimeBegin ="06:00:00";
		String dayTimeEnd ="18:00:00";
		
		long totalDayTime =0;

		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();

		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//�غ�ά�ȵ�����wifi����ͳ��
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpTime = UtilForMe.timeStampToDate(tmpTimeStamp, "HH:mm:ss");		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(dayTimeBegin.compareToIgnoreCase(tmpTime)<=0&&dayTimeEnd.compareToIgnoreCase(tmpTime)>=0){
					totalDayTime++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //��¼ÿ�����а��춨λʱ��					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpDayLong = oneClusterPresentDay.get(clusterIDTmp);
			
			if(tmpDayLong == null){
				tmpDayLong = new Long(0);
			}
			
			double ratioDayTime = 0;
			if(totalDayTime!=0){
				ratioDayTime =  (tmpDayLong.doubleValue())/totalDayTime;
			}
			
			//System.out.println("the cluster:ratioDayTime "+ clusterIDTmp +"\t"+ratioDayTime);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			//�����û����ڴص����ĵ㣬���뵽����list��
			statRes.get(clusterIDTmp).add(Double.toString(ratioDayTime));
			
		}	
		
	}
	
	
	
	/*
	 * @ͳ��ÿ���û�����ͬ�� ���ϵ����ӱ��� �û�ȫ�����϶�λ��������
	 * 
	 * ���㷽����
	 * �û����϶�λ��������ȥ�أ� ��Ϊy��
	 * ĳ�����У����ϵ�������Ϊx��
	 * 
	 * ���ϣ�0��00~6��00  18:00~23:59
	 * 
	 * �����x/y
	 * 
	 */
	public static void statRatioNightTime(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	

		String dayTimeBegin ="06:00:00";
		String dayTimeEnd ="18:00:00";
		
		long totalNightTime =0;

		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();

		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//�غ�ά�ȵ�����wifi����ͳ��
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpTime = UtilForMe.timeStampToDate(tmpTimeStamp, "HH:mm:ss");		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(!(dayTimeBegin.compareToIgnoreCase(tmpTime)<=0&&dayTimeEnd.compareToIgnoreCase(tmpTime)>=0)){
					totalNightTime++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //��¼ÿ�����а��춨λʱ��					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpNightLong = oneClusterPresentDay.get(clusterIDTmp);
			if(tmpNightLong == null){
				tmpNightLong = new Long(0);
			}
			
			double ratioNightTime = 0;
			if(totalNightTime!=0){
				ratioNightTime =  (tmpNightLong.doubleValue())/totalNightTime;
			}
			
			//System.out.println("the cluster:ratioDayTime "+ clusterIDTmp +"\t"+ratioNightTime);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(Double.toString(ratioNightTime));
			
		}			
	}
	
	
	
	/*
	 * @ͳ��ÿ���û�����ͬ�� ��ĩ�����ӱ��� �û�ȫ����ĩ��λ��������
	 * 
	 * ���㷽����
	 * �û���ĩ��λ��������ȥ�أ� ��Ϊy��
	 * ĳ�����У���ĩ��������Ϊx��
	 * 
	 * ���ϣ�0��00~6��00  18:00~23:59
	 * 
	 * �����x/y
	 * 
	 */
	public static void statRatioWeekend(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;			
		long totalWeekend =0;
		HashMap<String,Long> oneClusterPresentDay = new HashMap<String, Long>();			
		oneClusterPresentDay.clear();
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();			
			
			for (int i = 0; i < userPoiTmp.size(); i++) {
				//�غ�ά�ȵ�����wifi����ͳ��
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				boolean isWeekend = UtilForMe.timeStampIsWeekend(tmpTimeStamp);		
				
				//System.out.println("culstID:time:"+clusterIDTmp+"\t"+tmpTime);
				if(isWeekend){
					totalWeekend++;					
					if(!oneClusterPresentDay.containsKey(clusterIDTmp)){
						oneClusterPresentDay.put(clusterIDTmp, new Long(0));
					}
					Long tmpLong = oneClusterPresentDay.get(clusterIDTmp);
					oneClusterPresentDay.put(clusterIDTmp, ++tmpLong); //��¼ÿ������weekend��λʱ��					
				}
			}//end for
		}//end for

		for (Entry<String, ArrayList<String>> en2 : enties) {			
			clusterIDTmp = en2.getKey();			
			Long tmpWeekendLong = oneClusterPresentDay.get(clusterIDTmp);
			if(tmpWeekendLong == null){
				tmpWeekendLong = new Long(0);
			}
			
			double ratioWeekend = 0;
			if(totalWeekend!=0){
				ratioWeekend =  (tmpWeekendLong.doubleValue())/totalWeekend;
			}
			
			//System.out.println("the cluster:ratioWeekend "+ clusterIDTmp +"\t"+ratioWeekend);	
			
			if(!statRes.containsKey(clusterIDTmp)){
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(Double.toString(ratioWeekend));
			
		}			
	}
	
	
	/*
	 * @desc 
	 * ��һ��24h�ֳ�n��time slot;
	 * ����ÿ��time-slot�У��û�cuid��ÿ�����г��ֵ��ܵ���������Ϊy
	 * �����У��ڸ���ʱ����г��ֵĵ�<x1,x2,....>��ռ�������е�(y)�ı���
	 * 
	 * @para  slot  Ӧ�ÿ��Ա�24����
	 */
	
	

	public static void statRatioTimeSlot(HashMap<String, ArrayList<String>> statRes,int slot){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	


		long[] eachSlotCount = new long [slot];
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			//ÿ���س�ʼ��һ��
			for(int i=0;i<slot;i++){
				eachSlotCount[i] = 0;
			}
			
			long totalCount = userPoiTmp.size();//�˴������е�ĸ���
			
			for(int i=0;i<userPoiTmp.size();i++){
				
				parts = userPoiTmp.get(i).split("\t");

				String tmpTimeStamp = parts[2].trim();
				
				String tmpHour = UtilForMe.timeStampToDate(tmpTimeStamp, "HH");	
				
				//System.out.println("tmpHour: "+tmpHour);
				
				int eachSlot = 24/slot;  //ÿ�����У�ռ�õ�Сʱ����
				
				int index = Integer.parseInt(tmpHour)/eachSlot;
				eachSlotCount[index]++;				
			}
			
			//����ÿ��ʱ��۳��ֵĸô��е㣬��ռ�ô����е��ռ��
			for(int i=0;i<slot;i++){
				
				double ratioSlot = 0;
				if(totalCount!=0){
					ratioSlot = (double)eachSlotCount[i]/totalCount;
				}
				
				//System.out.println("eachslotcount: ratio "+eachSlotCount[i] +"\t" + ratioSlot);
				
				if(!statRes.containsKey(clusterIDTmp)){
					statRes.put(clusterIDTmp, new ArrayList<String>());
				}				
				statRes.get(clusterIDTmp).add(Double.toString(ratioSlot));			
			}			
		}
		
	}
	
	/*
	 * @desc 
	 * <clusterID1,(x,y,time,Wifi,clusterID,label?),(x,y,time,Wifi,clusterID,label?),...>
	 * 
	 * ��ÿ���أ����ϼһ�˾�ı�ǩ
	 * 
	 */
	
	public static void addLabel(HashMap<String, ArrayList<String>> statRes){
		
		Set<Entry<String, ArrayList<String>>> enties = userPoiGroupByCluster.entrySet();

		ArrayList<String> userPoiTmp = null;
		String clusterIDTmp = null;
		String[] parts;	
		
		for (Entry<String, ArrayList<String>> en : enties) {			
			clusterIDTmp = en.getKey();
			userPoiTmp = en.getValue();	
			
			parts = userPoiTmp.get(0).split("\t");
			
			String labelStr = null;
			
			//System.out.println("parts len: "+parts.length);
			
			if(parts.length==6){
				labelStr = parts[5];
			}else{
				labelStr = "";
			}		
			
			// System.out.println("eachslotcount: ratio "+eachSlotCount[i] +"\t"
			// + ratioSlot);
			
			//System.out.println(clusterIDTmp+"\t"+userPoiTmp);

			if (!statRes.containsKey(clusterIDTmp)) {
				statRes.put(clusterIDTmp, new ArrayList<String>());
			}
			statRes.get(clusterIDTmp).add(labelStr);				
		}
		
	}
	
	
	
	
	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String s1 = "11796016.000000	3011751.000000	1440900000	93351051973200	0	H";  //8.30 10:00:00
		String s2 = "11796018.000000	3011753.000000	1440471600	93351051973210	0	H";  //8.25 11:00:00
		String s8 = "11796018.000000	3011753.000000	1440586800	93351051973210	0	H";  //8.26 19:00:00
		
		String s3 = "11796018.000000	3011751.000000	1440640800	93351051973200	1	O";  // 8.27 10:00
		String s4 = "11796019.000000	3011741.000000	1440642600	93351051973100	1	O";  // 8.27 10:30:00
		String s5 = "11796019.000000	3011741.000000	1440586800	93351051973300	1	O";  // 8.26 19:00:00
		String s6 = "11796019.000000	3011741.000000	1440781201	93351051973100	1	O";  // 8.29 1:00:01
		String s7 = "11796019.000000	3011741.000000	1440460800	93351051973300	1	O";  // 8.25 8:00:00
		
		
		ArrayList<String> poiuser = new ArrayList<String>();
		
		poiuser.add(s1);
		poiuser.add(s2);
		poiuser.add(s3);
		poiuser.add(s4);
		poiuser.add(s5);
		poiuser.add(s6);
		poiuser.add(s7);
		poiuser.add(s8);
		
		
		
		FeatureStat.init(poiuser);		
		
		HashMap<String, ArrayList<String>> statRes = new HashMap<String, ArrayList<String>>() ;
		
		FeatureStat.statCenterCordinate(statRes);		
		
		FeatureStat.statWifiEntropy(statRes);
		
		FeatureStat.statRatioPresentDay(statRes);
		
		FeatureStat.statAvgStayTime(statRes);
		
		FeatureStat.statRatioDayTime(statRes);
		
		FeatureStat.statRatioNightTime(statRes);
		
		FeatureStat.statRatioWeekend(statRes);
		
		FeatureStat.statRatioTimeSlot(statRes,8);
		
		
		FeatureStat.addLabel(statRes);
		
		
		
		
		Set<Entry<String, ArrayList<String>>> enties = statRes.entrySet();
		
		ArrayList<String> userTmp=null;
		String clusterIDTmp=null;
		for(Entry<String, ArrayList<String>> en:enties){
			clusterIDTmp = en.getKey();
			userTmp = en.getValue();
			String feature = null;
			for(int i =0 ;i<userTmp.size();i++){
				
				//�γɸ�ʽ��clusterID		xx	yy  entropy
				if(i==0){
					feature = userTmp.get(i)+"\t";
				}else if(i==(userTmp.size()-1)){
					feature +=userTmp.get(i);
				}else{
					feature += userTmp.get(i)+"\t";
				}
				
				//System.out.println(clusterIDTmp+"\t"+userTmp.get(i));
			}
			
			System.out.println(clusterIDTmp+"\t"+feature);	
		}
		

		
		
		
	}

}
