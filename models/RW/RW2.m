function [ out ] = RW2( params,pnames,learning_data,post_learning_data,simulation,prior)
    ID=1; AGENT = 2; SESSION = 3; TRIAL=4 ;STATE=5; P1 = 6; P2=7; MAG=8; VAL = 9; INF = 10; CHOICE=11; OUTCOME=12; CF_OUTCOME=13;
  
    nb_states   = max(learning_data(:,STATE));
    
    beta = params(1);                                                             
    alpha_rpe   = params(2);       
    persev = params(3);
    
    last_choice = zeros(nb_states); 
  
    Q = zeros(nb_states,2);
    
    if(simulation)
        lik_all = zeros(length(learning_data),1);
        Q_all = zeros(length(learning_data),2);
    end
    
    p=0;
    if(prior)
        p = parameter_priors(params,pnames);
    end

    lik=0;    
    
    for i = 1:length(learning_data(:,TRIAL))
        
        if (~isnan(learning_data(i,AGENT)) && ~isnan(learning_data(i,CHOICE)))
            
            Qd = Q;
            if(last_choice(learning_data(i,STATE))>0)
                Qd(learning_data(i,STATE),last_choice(learning_data(i,STATE))) = Qd(learning_data(i,STATE),last_choice(learning_data(i,STATE))) + persev;
            end
            
            P = softmax(beta*Qd')';
            
            if(simulation)
                lik = lik + log(P(learning_data(i,STATE),learning_data(i,CHOICE)));
                lik_all(i)=lik;
                Q_all(i,:) = Q(learning_data(i,STATE),:);
                
                [choice,out,cf_out] = one_step(P(learning_data(i,STATE),:),learning_data(i,MAG),learning_data(i,VAL),learning_data(i,INF),learning_data(i,P1),learning_data(i,P2));
                
                learning_data(i,CHOICE)     = choice;
                learning_data(i,OUTCOME)    = out;
                learning_data(i,CF_OUTCOME) = cf_out;
                
            else
                
                lik = lik + log(P(learning_data(i,STATE),learning_data(i,CHOICE)));%compute likelihood
                
            end
            
            %private learning
            RPE =  learning_data(i,OUTCOME) - Q(learning_data(i,STATE),learning_data(i,CHOICE));
            Q(learning_data(i,STATE),learning_data(i,CHOICE)) = Q(learning_data(i,STATE),learning_data(i,CHOICE)) + alpha_rpe * RPE;
            
            RPE2 =  -learning_data(i,OUTCOME) - Q(learning_data(i,STATE),3-learning_data(i,CHOICE));
            Q(learning_data(i,STATE),3-learning_data(i,CHOICE)) = Q(learning_data(i,STATE),3-learning_data(i,CHOICE)) + alpha_rpe * RPE2;
            
            last_choice(learning_data(i,STATE))=learning_data(i,CHOICE);
        end
    end

    l= -lik;
    out=p+l;


    if(simulation)
        out = struct();
        out.sim_data=learning_data;
        out.pred_data = [lik_all lik_all Q_all];
    end
    
    
end

