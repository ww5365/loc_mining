package driver.loc;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

public class UtilForMe {

	public UtilForMe() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String secString  = "1446829200";
		
		String dString = timeStampToDate(secString, "yyyy-MM-dd");
		
		System.out.println("date:"+dString);
		
		
		String ddString = timeStampToDate(secString,"HH");
		
		System.out.println("date:"+ddString);
		
		String dayTimeBegin ="06:00:00";
		String dayTimeEnd ="18:00:00";
		
		if(dayTimeBegin.compareToIgnoreCase(ddString)<=0&&dayTimeEnd.compareToIgnoreCase(ddString)>=0){
			
			System.out.println("this is day time:"+ddString);
		}else{
			System.out.println("this is night time:"+ddString);
		}
		
		System.out.println("week is:" + timeStampIsWeekend(secString));

		
	}
	
	
	
	
	
	/*
	 * ��linuxʱ���ת��Ϊ���� ʱ�� �ַ���
	 */
	public static String timeStampToDate(String seconds,String format){
		
		if (seconds == null || seconds.isEmpty() || seconds.equals("null")) {
			return "";
		}
		if (format == null || format.isEmpty())
			format = "yyyy-MM-dd HH:mm:ss";
		SimpleDateFormat sdf = new SimpleDateFormat(format);
		return sdf.format(new Date(Long.valueOf(seconds + "000")));
		
	}
	
	
	
	/*
	 * ���룺linuxʱ���  ��  �ж��Ƿ�Ϊ��ĩ
	 */
	public static boolean timeStampIsWeekend(String seconds){
		
		if (seconds == null || seconds.isEmpty() || seconds.equals("null")) {
			return false;
		}
		
		//��ʱ���תΪ�������ͣ�Date
		Date curDate = new Date(Long.valueOf(seconds + "000"));
		
		//System.out.println("date:"+curDate);
		
		Calendar calendar = Calendar.getInstance();
		
		calendar.setTime(curDate);
		
		if(calendar.get(Calendar.DAY_OF_WEEK)==Calendar.SATURDAY||calendar.get(Calendar.DAY_OF_WEEK)==Calendar.SUNDAY){
			return true;
		}else{
			return false;
		}		
	}
	
	
	/*
	 * ����Double�����д洢�ĸ��ʷֲ�pi����
	 */
	
	public static double CalEntropy(ArrayList<Double> param){
		
		double res = -1;
		
		if(param.isEmpty()){
			return res;
		}
			
		double entropy = 0;
		
		
		for(int i=0;i<param.size();i++){			
			Double ratio = param.get(i);			
			entropy += ratio.doubleValue()*Math.log(ratio.doubleValue()) / Math.log(2);			
		}
		
		if(entropy == 0){
			res = entropy;	
		}else{
			res = -entropy;	
		}
	
		return res;		
	}
	

}
