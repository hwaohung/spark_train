attr1s = ["udp", "icmp", "tcp"]
attr1s_map = { attr1s[i]: i for i in range(len(attr1s)) }

attr2s = [
            "urp_i", "netbios_ssn", "Z39_50", "tim_i", "smtp",
            "domain", "private", "echo", "printer", "red_i",
            "eco_i", "sunrpc", "ftp_data", "urh_i", "pm_dump",
            "pop_3", "pop_2", "systat", "ftp", "uucp",
            "whois", "tftp_u", "netbios_dgm", "efs", "remote_job",
            "sql_net", "daytime", "ntp_u", "finger", "ldap",
            "netbios_ns", "kshell", "iso_tsap", "ecr_i", "nntp",
            "http_2784", "shell", "domain_u", "uucp_path", "courier",
            "exec", "aol", "netstat", "telnet", "gopher",
            "rje", "hostnames", "link", "ssh", "http_443",
            "csnet_ns", "X11", "IRC", "harvest", "imap4",
            "supdup", "name", "nnsp", "mtp", "http",
            "bgp", "ctf", "klogin", "vmnet", "time",
            "discard", "login", "auth", "other", "http_8001"
         ]
attr2s_map = { attr2s[i]: i for i in range(len(attr2s)) }

attr3s = [
            "OTH", "RSTR", "S3", "S2", "S1",
            "S0", "RSTOS0", "REJ", "SH", "RSTO",
            "SF"
         ]
attr3s_map = { attr3s[i]: i for i in range(len(attr3s)) }

labels = [ "back", "buffer_overflow", "ftp_write", "guess_passwd", "imap", "ipsweep",
           "land", "loadmodule", "multihop", "neptune", "nmap", "normal",
           "perl", "phf", "pod", "portsweep", "rootkit", "satan",
           "smurf", "spy", "teardrop", "warezclient", "warezmaster"
         ]
label_map = { labels[i]: i for i in range(len(labels)) }
